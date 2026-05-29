ProbabilisticCircuit <- setRefClass("ProbabilisticCircuit",
  fields = list(
    window_size  = "integer",
    ranks        = "integer",
    is_log_probs = "logical"
  ),
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      window_size  <<- as.integer(window_size)
      ranks        <<- as.integer(ranks)
      is_log_probs <<- FALSE
    },
    inject_head = function(model) { stop("Not implemented") },
    forward = function(model, hidden_states) { stop("Not implemented") },
    get_draft_probs = function(model, embeddings) { stop("Not implemented") },
    generate_draft = function(probs, batch_idx = 1L) { stop("Not implemented") },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) { stop("Not implemented") }
  )
)

HMMCircuit <- setRefClass("HMMCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- TRUE
    },
    inject_head = function(model) {
      ed <- as.integer(model$embed_dim)
      vs <- as.integer(model$vocab_size)
      layers <- list(
        init_gate   = nn$Linear(ed, ranks),
        emissions   = nn$Linear(ed, as.integer(window_size * ranks * vs)),
        transitions = nn$Linear(ed, as.integer((window_size - 1L) * ranks * ranks))
      )
      model$heads$update(layers)
      nn$init$zeros_(model$heads[["init_gate"]]$weight)
      nn$init$zeros_(model$heads[["init_gate"]]$bias)
      nn$init$normal_(model$heads[["transitions"]]$weight, mean = 0.0, std = 1e-4)
      bias_tensor <- torch$zeros(window_size - 1L, ranks, ranks)
      bias_tensor$diagonal(dim1 = -2L, dim2 = -1L)$fill_(5.0)
      reticulate::with(torch$no_grad(), {
        model$heads[["transitions"]]$bias$copy_(bias_tensor$flatten())
        stp_weight <- model$backbone$lm_head$weight$detach()$clone()
        H <- as.integer(stp_weight$shape[1])
        emission_init <- stp_weight$unsqueeze(0L)$unsqueeze(0L)$
          expand(window_size, ranks, -1L, -1L)$
          reshape(-1L, H)
        model$heads[["emissions"]]$weight$copy_(emission_init)
        nn$init$zeros_(model$heads[["emissions"]]$bias)
      })
      invisible(model)
    },
    forward = function(model, hidden_states) {
      batch_size <- as.integer(hidden_states$shape[0])
      seq_len    <- as.integer(hidden_states$shape[1])
      vocab_size <- as.integer(model$vocab_size)
      log_alpha <- F_$log_softmax(model$heads[["init_gate"]](hidden_states), dim = -1L)
      flat_emiss <- model$heads[["emissions"]](hidden_states)
      log_emiss <- F_$log_softmax(
        flat_emiss$view(batch_size, seq_len, window_size, ranks, vocab_size), dim = -1L
      )
      flat_trans <- model$heads[["transitions"]](hidden_states)
      log_trans <- F_$log_softmax(
        flat_trans$view(batch_size, seq_len, window_size - 1L, ranks, ranks), dim = -1L
      )
      log_marginal_probs <- list()
      for (t in seq(0L, window_size - 1L)) {
        curr_emission <- log_emiss$select(2L, t)
        step_prob <- torch$logsumexp(log_alpha$unsqueeze(-1L) + curr_emission, dim = 2L)
        log_marginal_probs[[t + 1L]] <- step_prob
        if (t < window_size - 1L) {
          curr_trans <- log_trans$select(2L, t)
          log_alpha <- torch$logsumexp(log_alpha$unsqueeze(-1L) + curr_trans, dim = 2L)
        }
      }
      torch$stack(log_marginal_probs, dim = 2L)
    },
    get_draft_probs = function(model, embeddings) {
      reticulate::with(torch$no_grad(), {
        ndims <- length(embeddings$shape)
        if (ndims == 3) {
          last_emb <- embeddings$select(1L, -1L)$unsqueeze(1L)
        } else if (ndims == 2) {
          last_emb <- embeddings$select(0L, -1L)$unsqueeze(0L)$unsqueeze(0L)
        } else {
          last_emb <- embeddings
        }

        batch_size <- as.integer(last_emb$size(0L))
        vocab_size <- as.integer(model$vocab_size)
        
        init_probs <- F_$softmax(model$heads[["init_gate"]](last_emb)$squeeze(1L), dim = -1L)
        flat_emiss  <- model$heads[["emissions"]](last_emb)
        emiss_probs <- F_$softmax(
          flat_emiss$view(as.integer(batch_size), as.integer(window_size), as.integer(ranks), as.integer(vocab_size)), dim = -1L
        )
        flat_trans  <- model$heads[["transitions"]](last_emb)
        trans_probs <- F_$softmax(
          flat_trans$view(batch_size, window_size - 1L, ranks, ranks), dim = -1L
        )
        list(
          init  = init_probs$cpu()$numpy(),
          emiss = emiss_probs$cpu()$numpy(),
          trans = trans_probs$cpu()$numpy()
        )
      })
    },
    generate_draft = function(probs, batch_idx = 1L) {
      init_probs  <- probs$init
      emiss_probs <- probs$emiss
      trans_probs <- probs$trans
      draft_tokens  <- integer(window_size)
      ranks_indices <- seq_len(dim(init_probs)[2])
      vocab_indices <- seq_len(dim(emiss_probs)[4])
      z_t <- sample(ranks_indices, 1, prob = init_probs[batch_idx, ])
      for (t in seq_len(window_size)) {
        x_t <- sample(vocab_indices, 1, prob = emiss_probs[batch_idx, t, z_t, ])
        draft_tokens[t] <- x_t - 1L
        if (t < window_size) {
          z_t <- sample(ranks_indices, 1, prob = trans_probs[batch_idx, t, z_t, ])
        }
      }
      draft_tokens
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      ws <- length(draft_tokens)
      q_values <- numeric(ws)
      current_token <- draft_tokens[1] + 1
      alpha <- probs$init[batch_idx, ] * probs$emiss[batch_idx, 1, , current_token]
      q_values[1] <- sum(alpha)
      if (ws > 1) {
        for (t in 2:ws) {
          trans_matrix <- probs$trans[batch_idx, t - 1, , ]
          alpha <- as.vector(alpha %*% trans_matrix)
          current_token <- draft_tokens[t] + 1
          alpha <- alpha * probs$emiss[batch_idx, t, , current_token]
          q_values[t] <- sum(alpha)
        }
      }
      return(q_values)
    },
    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      alpha <- probs$init[batch_idx, ]
      if (step > 1) {
        for (t in 2:step) {
          trans_matrix <- probs$trans[batch_idx, t - 1, , ]
          alpha <- as.vector(alpha %*% trans_matrix)
        }
      }
      emiss_matrix <- probs$emiss[batch_idx, step, , ]
      vocab_dist <- as.vector(alpha %*% emiss_matrix)
      vocab_dist <- vocab_dist / sum(vocab_dist)
      return(vocab_dist)
    }
  )
)

FFCircuit <- setRefClass("FFCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- FALSE
    },
    inject_head = function(model) {
      layers <- list()
      for (i in seq_len(window_size)) {
        layers[[sprintf("emission_%d", i)]] <- nn$Linear(
          as.integer(model$embed_dim), as.integer(model$vocab_size)
        )
      }
      model$heads$update(layers)
      reticulate::with(torch$no_grad(), {
        stp_weight <- model$backbone$lm_head$weight$detach()$clone()
        for (i in seq_len(window_size)) {
          key <- sprintf("emission_%d", i)
          model$heads[[key]]$weight$copy_(stp_weight)
          nn$init$zeros_(model$heads[[key]]$bias)
        }
      })
      invisible(model)
    },
    forward = function(model, hidden_states) {
      if (length(hidden_states$size()) == 3 && hidden_states$size(1L) == 1L) {
        hidden_states <- hidden_states$squeeze(1L)
      }
      logits_list <- list()
      for (i in seq_len(window_size)) {
        key <- sprintf("emission_%d", i)
        logits_list[[i]] <- model$heads[[key]](hidden_states)
      }
      stack_dim <- if (length(hidden_states$size()) == 3) 2L else 1L
      torch$stack(logits_list, dim = stack_dim)
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      ws <- length(draft_tokens)
      q_values <- numeric(ws)
      current_q <- 1.0
      p_dims <- dim(probs)
      
      # Troviamo l'indice dell'ultimo token nella sequenza
      seq_idx <- if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      
      for (t in seq_len(ws)) {
        token_id <- draft_tokens[t] + 1
        if (length(p_dims) == 4) {
          # Shape: [Batch, Seq, Window, Vocab] -> prendiamo l'ultimo Seq, step t
          prob_t <- probs[batch_idx, seq_idx, t, token_id]
        } else if (length(p_dims) == 3) {
          # Shape: [Seq, Window, Vocab] -> prendiamo l'ultimo Seq, step t
          prob_t <- probs[seq_idx, t, token_id]
        } else if (length(p_dims) == 2) {
          # Shape: [Window, Vocab]
          prob_t <- probs[t, token_id]
        } else {
          prob_t <- probs[token_id]
        }
        current_q <- current_q * prob_t
        q_values[t] <- current_q
      }
      return(q_values)
    },
    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      p_dims <- dim(probs)
      seq_idx <- if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      
      if (length(p_dims) == 4) {
        return(as.vector(probs[batch_idx, seq_idx, step, ]))
      } else if (length(p_dims) == 3) {
        return(as.vector(probs[seq_idx, step, ]))
      } else {
        return(as.vector(probs))
      }
    },
    get_draft_probs = function(model, embeddings) {
      with(torch$no_grad(), {
        logits <- .self$forward(model, embeddings)
        probs <- F_$softmax(logits, dim = -1L)
        res <- reticulate::py_to_r(probs$cpu()$numpy())
        # cat(sprintf("  [DEBUG PC] Shape Probs: %s\n", paste(dim(res), collapse="x")))
        return(res)
      })
    },
    generate_draft = function(probs, batch_idx = 1L) {
      p_dims <- dim(probs)
      # cat(sprintf("  [DEBUG PC] generate_draft input shape: %s\n", paste(p_dims, collapse="x")))
      
      draft_tokens <- integer(window_size)
      
      # Troviamo l'indice dell'ultimo token nella sequenza (dimensione 2 se 4D, dimensione 1 se 3D)
      seq_idx <- if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      
      for (t in seq_len(window_size)) {
        if (length(p_dims) == 4) {
          # Shape: [Batch, Seq, Window, Vocab] -> prendiamo l'ultimo Seq
          row_t <- probs[batch_idx, seq_idx, t, ]
        } else if (length(p_dims) == 3) {
          # Shape: [Seq, Window, Vocab] -> prendiamo l'ultimo Seq
          row_t <- probs[seq_idx, t, ]
        } else if (length(p_dims) == 2) {
          # Shape: [Window, Vocab]
          row_t <- probs[t, ]
        } else {
          row_t <- probs
        }
        draft_tokens[t] <- which.max(row_t) - 1L
      }
      draft_tokens
    }

  )
)

CPCircuit <- setRefClass("CPCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- TRUE
    },
    inject_head = function(model) {
      ed <- as.integer(model$embed_dim)
      vs <- as.integer(model$vocab_size)
      layers <- list(
        gate           = nn$Linear(ed, ranks),
        emission_projs = nn$Linear(ed, as.integer(ranks * window_size * vs))
      )
      model$heads$update(layers)
      nn$init$zeros_(model$heads[["gate"]]$weight)
      nn$init$zeros_(model$heads[["gate"]]$bias)
      reticulate::with(torch$no_grad(), {
        stp_weight <- model$backbone$lm_head$weight$detach()$clone()
        H <- as.integer(stp_weight$shape[1])
        emission_init <- stp_weight$unsqueeze(0L)$unsqueeze(0L)$
          expand(ranks, window_size, -1L, -1L)$
          reshape(-1L, H)
        model$heads[["emission_projs"]]$weight$copy_(emission_init)
        nn$init$zeros_(model$heads[["emission_projs"]]$bias)
      })
    },
    forward = function(model, hidden_states) {

      gate_logits <- model$heads$gate(hidden_states)
      log_weights <- F_$log_softmax(gate_logits, dim = -1L) # [B, S, R]
      
      flat_emissions <- model$heads$emission_projs(hidden_states)
      batch_size <- as.integer(hidden_states$shape[0])
      seq_len    <- as.integer(hidden_states$shape[1])
      
      emissions <- flat_emissions$view(
        c(batch_size, seq_len, as.integer(ranks), as.integer(window_size), as.integer(model$vocab_size))
      )
      log_token_probs <- F_$log_softmax(emissions, dim = -1L) # [B, S, R, W, V]
      
      # Bug 2 Fix: Allineamento pesi per broadcasting [B, S, R, 1, 1]
      # Usiamo -1L per aggiungere dimensioni alla fine in modo sicuro
      log_weights_exp <- log_weights$unsqueeze(-1L)$unsqueeze(-1L)
      
      # Marginalizzazione lungo i Ranks (Dimensione 2 in 0-based indexing: B=0, S=1, R=2)
      torch$logsumexp(log_weights_exp + log_token_probs, dim = 2L)
    },
    get_draft_probs = function(model, embeddings) {
      with(torch$no_grad(), {
        # Bug 2 Fix: Selezione ultimo token (Seq è dim 1 in 0-based)
        last_emb <- embeddings$select(1L, -1L)$unsqueeze(1L)
        
        gate_logits <- model$heads$gate(last_emb)
        gate_probs  <- F_$softmax(gate_logits, dim = -1L)$squeeze(1L)
        
        flat_emissions <- model$heads$emission_projs(last_emb)
        batch_size <- as.integer(last_emb$shape[0])
        
        emissions <- flat_emissions$view(
          c(batch_size, as.integer(ranks), as.integer(window_size), as.integer(model$vocab_size))
        )
        emiss_probs <- F_$softmax(emissions, dim = -1L)
        
        # Bug 1 Fix: Restituiamo componenti separate per Campionamento Ancestrale
        list(
          gate = reticulate::py_to_r(gate_probs$cpu()$numpy()),
          emiss = reticulate::py_to_r(emiss_probs$cpu()$numpy())
        )
      })
    },
    generate_draft = function(probs, batch_idx = 1L) {
      # Bug 1 Fix: Ancestral Sampling (commit ad una specifica traiettoria Z)
      gate_probs  <- probs$gate
      emiss_probs <- probs$emiss
      
      # 1. Campionamento del Rank (Z) basato sul Gate
      z_dist  <- gate_probs[batch_idx, ]
      z_sampled_idx <- sample(seq_along(z_dist), 1, prob = z_dist)
      
      # 2. Campionamento dei token (X) condizionati al Rank Z scelto
      draft_tokens <- integer(window_size)
      for (t in seq_len(window_size)) {
        # emiss_probs: [Batch, Rank, Window, Vocab]
        row_t <- emiss_probs[batch_idx, z_sampled_idx, t, ]
        # Usiamo sample() per mantenere la natura stocastica del modello di miscela
        draft_tokens[t] <- sample(seq_along(row_t) - 1, 1, prob = row_t)
      }
      draft_tokens
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      ws <- length(draft_tokens)
      q_values <- numeric(ws)
      
      # P(x_1...x_k) = sum_z P(z) * product_{t=1}^k P(x_t | z)
      # gate: [Batch, Rank], emiss: [Batch, Rank, Window, Vocab]
      gate <- probs$gate[batch_idx, ] # [R]
      
      # Inizializziamo alpha come P(z)
      alpha <- gate
      
      for (t in seq_len(ws)) {
        token_id <- draft_tokens[t] + 1
        # Aggiorniamo alpha con P(x_t | z)
        alpha <- alpha * probs$emiss[batch_idx, , t, token_id]
        # La probabilità del prefisso è la somma su z
        q_values[t] <- sum(alpha)
      }
      return(q_values)
    },
    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      # P(x_step) = sum_z P(z) * P(x_step | z)
      gate <- probs$gate[batch_idx, ] # [R]
      emiss_step <- probs$emiss[batch_idx, , step, ] # [R, V]
      
      # Prodotto riga per riga e poi somma sulle colonne (Ranks)
      marginal_dist <- as.vector(gate %*% emiss_step)
      return(marginal_dist)
    }


  )
)

create_circuit <- function(type, window_size = 8L, ranks = 32L) {
  switch(tolower(type),
    "hmm" = HMMCircuit$new(window_size = window_size, ranks = ranks),
    "ff"  = FFCircuit$new(window_size = window_size, ranks = ranks),
    "cp"  = CPCircuit$new(window_size = window_size, ranks = ranks),
    stop(sprintf("Tipo di circuito sconosciuto: '%s'. Usa 'hmm', 'ff', o 'cp'.", type))
  )
}
