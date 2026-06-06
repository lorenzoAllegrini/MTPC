ProbabilisticCircuit = setRefClass("ProbabilisticCircuit",
  fields = list(
    window_size  = "integer",
    ranks        = "integer",
    is_log_probs = "logical"
  ),
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      # initializes the base probabilistic circuit with window size and ranks
      window_size  <<- as.integer(window_size)
      ranks        <<- as.integer(ranks)
      is_log_probs <<- FALSE
    },
    inject_head = function(model) { 
      # injects prediction head layers into the wrapper model
      stop("Not implemented") 
    },
    forward = function(model, hidden_states) { 
      # computes joint sequence log probabilities for the given hidden states
      stop("Not implemented") 
    },
    get_draft_probs = function(model, embeddings) { 
      # extracts draft prediction probabilities from hidden states
      stop("Not implemented") 
    },
    generate_draft = function(probs, batch_idx = 1L) { 
      # generates draft tokens from draft probabilities
      stop("Not implemented") 
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) { 
      # computes cumulative joint prefix probabilities for draft tokens
      stop("Not implemented") 
    },
    get_conditional_dist = function(probs, accepted_prefix, batch_idx = 1L) { 
      # computes the conditional distribution for the next token given an accepted prefix
      stop("Not implemented") 
    }
  )
)

HMMCircuit = setRefClass("HMMCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      # initializes the inhomogeneous hmm circuit with window size and ranks
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- TRUE
    },
    inject_head = function(model) {
      # injects and initializes inhomogeneous hmm head layers (init, emissions, transitions) into the model
      
      ed = as.integer(model$embed_dim)
      vs = as.integer(model$vocab_size)
      layers = list(
        sum_unit_omega_init = nn$Linear(ed, ranks), #initial distribution probability of latent states
        input_units_phi = nn$Linear(ed, as.integer(window_size * ranks * vs)),
        sum_unit_omega_transitions = nn$Linear(ed, as.integer((window_size - 1L) * ranks * ranks)) #inhomogeneous hmm!
      )
      #call to ModuleDict
      model$heads$update(layers)

      # state gates initialized to zero, will be changed in a few lines
      nn$init$zeros_(model$heads[["sum_unit_omega_init"]]$weight)
      nn$init$zeros_(model$heads[["sum_unit_omega_init"]]$bias)

      #transition matrix are initialized as Identity with a bit of gaussian noise
      nn$init$normal_(model$heads[["sum_unit_omega_transitions"]]$weight, mean = 0.0, std = 1e-4)
      # bias_tensor shape: [window_size - 1, ranks, ranks]
      bias_tensor = torch$zeros(window_size - 1L, ranks, ranks)
      # here we do not put 1 in the diagonal because we are working in log space
      bias_tensor$diagonal(dim1 = -2L, dim2 = -1L)$fill_(5.0)

      with(torch$no_grad(), {
        #insert bias vector
        model$heads[["sum_unit_omega_transitions"]]$bias$copy_(bias_tensor$flatten())
        
        # retrieve stp weights
        if (!is.null(model$backbone$lm_head)) {
          stp_weight = model$backbone$lm_head$weight$detach()$clone()
        } else {
          stp_weight = model$backbone$get_output_embeddings()$weight$detach()$clone()
        }

        # stp_weight shape: [vocab_size, embed_dim]
        H = as.integer(stp_weight$shape[1])

        # emission_init shape: [window_size * ranks * vocab_size, embed_dim]
        emission_init = stp_weight$unsqueeze(0L)$unsqueeze(0L)$ #[vocab_size, embed_dim] -> [1, 1, vocab_size, embed_dim]
          expand(window_size, ranks, -1L, -1L)$ # clones into the dimensions with value 1, shape: [window_size, ranks, vocab_size, embed_dim]
          reshape(-1L, H)
        model$heads[["input_units_phi"]]$weight$copy_(emission_init)
        nn$init$zeros_(model$heads[["input_units_phi"]]$bias)
      })
      invisible(model)
    },
    forward = function(model, hidden_states) {
      # computes inhomogeneous hmm log marginal probabilities for speculative decoding
      
      # hidden_states shape: [batch_size, seq_len, embed_dim]
      batch_size = as.integer(hidden_states$shape[0])
      seq_len    = as.integer(hidden_states$shape[1])
      vocab_size = as.integer(model$vocab_size)
      
      # log_alpha shape: [batch_size, seq_len, ranks]
      log_alpha = F_$log_softmax(model$heads[["sum_unit_omega_init"]](hidden_states), dim = -1L)
      flat_emiss = model$heads[["input_units_phi"]](hidden_states)
      # log_emiss shape: [batch_size, seq_len, window_size, ranks, vocab_size]
      log_emiss = F_$log_softmax(
        flat_emiss$view(batch_size, seq_len, window_size, ranks, vocab_size), dim = -1L
      )
      flat_trans = model$heads[["sum_unit_omega_transitions"]](hidden_states)
      # log_trans shape: [batch_size, seq_len, window_size - 1, ranks, ranks]
      log_trans = F_$log_softmax(
        flat_trans$view(batch_size, seq_len, window_size - 1L, ranks, ranks), dim = -1L
      )
      log_marginal_probs = list()
      for (t in seq(0L, window_size - 1L)) {
        # curr_emission shape: [batch_size, seq_len, ranks, vocab_size]
        curr_emission = log_emiss$select(2L, t)
        # step_prob shape: [batch_size, seq_len, vocab_size]
        step_prob = torch$logsumexp(log_alpha$unsqueeze(-1L) + curr_emission, dim = 2L)
        log_marginal_probs[[t + 1L]] = step_prob
        if (t < window_size - 1L) {
          # curr_trans shape: [batch_size, seq_len, ranks, ranks]
          curr_trans = log_trans$select(2L, t)
          # log_alpha shape: [batch_size, seq_len, ranks]
          log_alpha = torch$logsumexp(log_alpha$unsqueeze(-1L) + curr_trans, dim = 2L)
        }
      }
      # return shape: [batch_size, seq_len, window_size, vocab_size]
      torch$stack(log_marginal_probs, dim = 2L)
    },
    get_draft_probs = function(model, embeddings) {
      # retrieves and formats draft transition, emission, and initial probabilities
      
      with(torch$no_grad(), {
        ndims = length(embeddings$shape)
        if (ndims == 3) {
          # last_emb shape: [batch_size, 1, embed_dim]
          last_emb = embeddings$select(1L, -1L)$unsqueeze(1L)
        } else if (ndims == 2) {
          # last_emb shape: [1, 1, embed_dim]
          last_emb = embeddings$select(0L, -1L)$unsqueeze(0L)$unsqueeze(0L)
        } else {
          last_emb = embeddings
        }

        batch_size = as.integer(last_emb$size(0L))
        vocab_size = as.integer(model$vocab_size)
        
        # init_probs shape: [batch_size, ranks]
        init_probs = F_$softmax(model$heads[["sum_unit_omega_init"]](last_emb)$squeeze(1L), dim = -1L)
        flat_emiss  = model$heads[["input_units_phi"]](last_emb)
        # emiss_probs shape: [batch_size, window_size, ranks, vocab_size]
        emiss_probs = F_$softmax(
          flat_emiss$view(as.integer(batch_size), as.integer(window_size), as.integer(ranks), as.integer(vocab_size)), dim = -1L
        )
        flat_trans  = model$heads[["sum_unit_omega_transitions"]](last_emb)
        # trans_probs shape: [batch_size, window_size - 1, ranks, ranks]
        trans_probs = F_$softmax(
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
      # samples a draft sequence of tokens from hmm probabilities
      z = sample(seq_len(dim(probs$init)[2]), 1, prob = probs$init[batch_idx, ])
      draft_tokens = integer(window_size)
      for (t in seq_len(window_size)) {
        draft_tokens[t] = sample(seq_len(dim(probs$emiss)[4]), 1, prob = probs$emiss[batch_idx, t, z, ]) - 1L
        if (t < window_size) z = sample(seq_len(dim(probs$trans)[3]), 1, prob = probs$trans[batch_idx, t, z, ])
      }
      draft_tokens
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      # computes joint prefix probabilities for hmm drafted tokens
      ws = length(draft_tokens)
      alpha = probs$init[batch_idx, ] * probs$emiss[batch_idx, 1, , draft_tokens[1] + 1]
      q_values = numeric(ws)
      q_values[1] = sum(alpha)
      if (ws > 1) {
        for (t in 2:ws) {
          alpha = as.vector(alpha %*% probs$trans[batch_idx, t - 1, , ]) * probs$emiss[batch_idx, t, , draft_tokens[t] + 1]
          q_values[t] = sum(alpha)
        }
      }
      q_values
    },
    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      # computes the full vocabulary distribution at a given step of the draft
      alpha = probs$init[batch_idx, ]
      if (step > 1) {
        for (t in 2:step) alpha = as.vector(alpha %*% probs$trans[batch_idx, t - 1, , ])
      }
      v = as.vector(alpha %*% probs$emiss[batch_idx, step, , ])
      v / sum(v)
    },
    get_conditional_dist = function(probs, accepted_prefix, batch_idx = 1L) {
      # computes the conditional distribution for the next token given an accepted prefix
      s = length(accepted_prefix)
      alpha = probs$init[batch_idx, ]
      if (s > 0) {
        alpha = alpha * probs$emiss[batch_idx, 1, , accepted_prefix[1] + 1]
        if (s > 1) {
          for (t in 2:s) {
            alpha = as.vector(alpha %*% probs$trans[batch_idx, t - 1, , ]) * probs$emiss[batch_idx, t, , accepted_prefix[t] + 1]
          }
        }
        alpha = as.vector(alpha %*% probs$trans[batch_idx, s, , ])
      }
      emiss_slice = probs$emiss[batch_idx, s + 1, , ]
      q_dist = as.numeric(matrix(alpha, nrow = 1) %*% emiss_slice)
      q_dist / sum(alpha)
    }
  )
)

FFCircuit = setRefClass("FFCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      # initializes the feed forward circuit with window size and ranks
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- FALSE
    },
    inject_head = function(model) {
      # injects and initializes independent feed forward speculative prediction heads
      layers = list()
      for (i in seq_len(window_size)) {
        layers[[sprintf("input_units_phi_%d", i)]] = nn$Linear(
          as.integer(model$embed_dim), as.integer(model$vocab_size)
        )
      }
      model$heads$update(layers)
      with(torch$no_grad(), {
        if (!is.null(model$backbone$lm_head)) {
          # stp_weight shape: [vocab_size, embed_dim]
          stp_weight = model$backbone$lm_head$weight$detach()$clone()
        } else {
          # stp_weight shape: [vocab_size, embed_dim]
          stp_weight = model$backbone$get_output_embeddings()$weight$detach()$clone()
        }
        for (i in seq_len(window_size)) {
          key = sprintf("input_units_phi_%d", i)
          model$heads[[key]]$weight$copy_(stp_weight)
          nn$init$zeros_(model$heads[[key]]$bias)
        }
      })
      invisible(model)
    },
    forward = function(model, hidden_states) {
      # computes independent logits for each future step and stacks them
      
      # hidden_states shape: [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
      if (length(hidden_states$size()) == 3 && hidden_states$size(1L) == 1L) {
        hidden_states = hidden_states$squeeze(1L)
      }
      logits_list = list()
      for (i in seq_len(window_size)) {
        key = sprintf("input_units_phi_%d", i)
        logits_list[[i]] = model$heads[[key]](hidden_states)
      }
      stack_dim = if (length(hidden_states$size()) == 3) 2L else 1L
      # return shape: [batch_size, seq_len, window_size, vocab_size] or [batch_size, window_size, vocab_size]
      torch$stack(logits_list, dim = stack_dim)
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      # computes prefix joint probabilities for independent feed forward heads
      p_dims = dim(probs)
      seq_idx = if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      p_t = sapply(seq_along(draft_tokens), function(t) {
        if (length(p_dims) == 4) probs[batch_idx, seq_idx, t, draft_tokens[t] + 1]
        else if (length(p_dims) == 3) probs[seq_idx, t, draft_tokens[t] + 1]
        else if (length(p_dims) == 2) probs[t, draft_tokens[t] + 1]
        else probs[draft_tokens[t] + 1]
      })
      cumprod(p_t)
    },
    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      # returns the vocabulary distribution for a specific speculative step
      p_dims = dim(probs)
      seq_idx = if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      if (length(p_dims) == 4) {
        return(as.vector(probs[batch_idx, seq_idx, step, ]))
      } else if (length(p_dims) == 3) {
        return(as.vector(probs[seq_idx, step, ]))
      } else {
        return(as.vector(probs))
      }
    },
    get_draft_probs = function(model, embeddings) {
      # computes speculative token probabilities for all steps from hidden embeddings
      with(torch$no_grad(), {
        logits = .self$forward(model, embeddings)
        probs = F_$softmax(logits, dim = -1L)
        res = reticulate::py_to_r(probs$cpu()$numpy())
        return(res)
      })
    },
    generate_draft = function(probs, batch_idx = 1L) {
      # generates draft tokens by taking argmax at each window step
      p_dims = dim(probs)
      seq_idx = if (length(p_dims) == 4) p_dims[2] else if (length(p_dims) == 3) p_dims[1] else 1
      sub_prob = if (length(p_dims) == 4) probs[batch_idx, seq_idx, , ] else if (length(p_dims) == 3) probs[seq_idx, , ] else if (length(p_dims) == 2) probs else matrix(probs, nrow = window_size, byrow = TRUE)
      apply(sub_prob, 1, which.max) - 1L
    },
    get_conditional_dist = function(probs, accepted_prefix, batch_idx = 1L) {
      # retrieves the next-step conditional vocabulary distribution
      step = length(accepted_prefix) + 1L
      .self$get_full_vocab_dist(probs, step, batch_idx)
    }
  )
)

CPCircuit = setRefClass("CPCircuit",
  contains = "ProbabilisticCircuit",
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      # initializes the cp circuit with window size and ranks
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- TRUE
    },

    inject_head = function(model) {
      # injects and initialize cp circuit heads (in particular the weights of gate and emissions) into the llm, takes as input an instance of LLMWrapper
      
      ed = as.integer(model$embed_dim)
      vs = as.integer(model$vocab_size)

      #trainable parameters
      layers = list(
        sum_unit_omega  = nn$Linear(ed, ranks),
        input_units_phi = nn$Linear(ed, as.integer(ranks * window_size * vs))
        #note: the product units do not have trainable weights
      )

      #update the moduledict of the LLMWrapper class 
      model$heads$update(layers)

      #initialize to zero the linear weights from embedding
      nn$init$zeros_(model$heads[["sum_unit_omega"]]$weight)
      nn$init$zeros_(model$heads[["sum_unit_omega"]]$bias)

      #initialization of the input weights so that they start as multiple copies (as the window size) of the output matrix of a stp llm
      with(torch$no_grad(), {

        #take the shape of the backbone embeddings
        if (!is.null(model$backbone$lm_head)) {
          stp_weight = model$backbone$lm_head$weight$detach()$clone()
        } else {
          stp_weight = model$backbone$get_output_embeddings()$weight$detach()$clone()
        }
        #stp_weight shape: [vocabulary size, embedding_size] 
        H = as.integer(stp_weight$shape[1])

        emission_init = stp_weight$unsqueeze(0L)$unsqueeze(0L)$ # [vocabulary size, embedding_size] -> [1, vocabulary size, embedding_size] -> [1, 1, vocabulary size, embedding_size] 
          expand(ranks, window_size, -1L, -1L)$ # [1, 1, vocabulary size, embedding_size] -> [n_ranks, window_size, vocabulary size, embedding_size] 
          reshape(-1L, H) # [n_ranks, window_size, vocabulary size, embedding_size] -> [n_ranks * window_size * vocabulary size, embedding_size] 

        model$heads[["input_units_phi"]]$weight$copy_(emission_init)
        nn$init$zeros_(model$heads[["input_units_phi"]]$bias)
      })
    },

     forward = function(model, hidden_states) {
      # takes as input an instance of LLMWrapper and the hidden states of the backbone, shape [batch_size, sequence_length, embedding_dim]
      gate_logits = model$heads$sum_unit_omega(hidden_states) # [batch_size, seq_len, ranks]
      log_weights = F_$log_softmax(gate_logits, dim = -1L)

      flat_emissions = model$heads$input_units_phi(hidden_states) #[batch_size, seq_len, ranks * window_size * vocab_size] 
      batch_size = as.integer(hidden_states$shape[0])
      seq_len    = as.integer(hidden_states$shape[1])
      
      emissions = flat_emissions$view(
        c(batch_size, seq_len, as.integer(ranks), as.integer(window_size), as.integer(model$vocab_size))
      ) 
      log_token_probs = F_$log_softmax(emissions, dim = -1L) # [batch_size, seq_len, ranks, window_size, vocab_size] 
      
      # expansion of the gate for broadcasting [batch_size, seq_len, ranks, 1, 1]
      log_weights_exp = log_weights$unsqueeze(-1L)$unsqueeze(-1L) 
      
      #weighted mixure of each latent category, 
      sum_unit = torch$logsumexp(log_weights_exp + log_token_probs, dim = 2L)
      
      return(sum_unit)
    },

    get_draft_probs = function(model, embeddings) {  
      # retrieves and formats draft gate and emission probabilities from last embedding
      with(torch$no_grad(), {
        last_emb = embeddings$select(1L, -1L)$unsqueeze(1L) # [batch_size, seq_len, embedding_dim] -> [batch_size, 1, embedding_dim]      

        gate_logits = model$heads$sum_unit_omega(last_emb) # [batch_size, 1, ranks]  
        gate_probs  = F_$softmax(gate_logits, dim = -1L)$squeeze(1L) # [batch_size, ranks]   

        flat_emissions = model$heads$input_units_phi(last_emb) # [batch_size, 1, ranks * window_size * vocab_size]   
        batch_size = as.integer(last_emb$shape[0])
        emissions = flat_emissions$view(
          c(batch_size, as.integer(ranks), as.integer(window_size), as.integer(model$vocab_size))
        )# [batch_size, ranks, window_size, vocab_size]     
        emiss_probs = F_$softmax(emissions, dim = -1L) # [batch_size, ranks, window_size, vocab_size]        

        list(
          gate  = as.matrix(gate_probs$cpu()$numpy()), # [batch_size, ranks]
          emiss = as.array(emiss_probs$cpu()$numpy()) # [batch_size, ranks, window_size, vocab_size]
        )
      })
    },
    generate_draft = function(probs, batch_idx = 1L) {
      # generates draft tokens using marginal argmax at each step
      sapply(seq_len(window_size), function(t) {
        dist = .self$get_full_vocab_dist(probs, t, batch_idx)
        which.max(dist) - 1L
      })
    },
    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      #computes the cumulative joint probability of the drafted tokens

      #probability of each latent state given the context
      alpha = probs$gate[batch_idx, ]

      q_values = numeric(length(draft_tokens))
      for (t in seq_along(draft_tokens)) {
        #probability of each drafted token given the state and the context
        alpha = alpha * probs$emiss[batch_idx, , t, draft_tokens[t] + 1]

        #comulative probability of the prefix sequence, marginalized over all states
        q_values[t] = sum(alpha)
      }
      q_values
    },

    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      # computes the full vocabulary distribution marginalized over latent states
      gate = probs$gate[batch_idx, ]
      emiss_step = probs$emiss[batch_idx, , step, ]
      marginal_dist = as.vector(gate %*% emiss_step)
      return(marginal_dist)
        },

    get_conditional_dist = function(probs, accepted_prefix, batch_idx = 1L) {
      # computes the conditional distribution for the next token given an accepted prefix
      s = length(accepted_prefix)
      alpha = probs$gate[batch_idx, ]
      if (s > 0) {
        for (t in 1:s) {
          alpha = alpha * probs$emiss[batch_idx, , t, accepted_prefix[t] + 1]
        }
      }
      emiss_slice = probs$emiss[batch_idx, , s + 1, ]
      q_dist = as.numeric(matrix(alpha, nrow = 1) %*% emiss_slice)
      denom = sum(alpha)
      if (denom > 0) q_dist / denom else q_dist
    }
  )
)

build_btree_topology = function(window_size) {
  # Balanced binary tree of latents over `window_size` token positions: recursively split a span
  # at floor(len/2) down to single-token leaves (MTPC-BTree, Fig. 2). 1-based indices.
  #   node_parent[k]  = parent internal-node index (0 for root)
  #   token_parent[i] = internal-node index that emits token position i
  # Internal nodes are numbered pre-order (parent index < child index).
  node_parent = integer(0)
  token_parent = integer(window_size)
  build = function(toks, parent) {
    node_parent[[length(node_parent) + 1L]] <<- parent
    idx = length(node_parent)
    h = length(toks) %/% 2L
    for (sub in list(toks[seq_len(h)], toks[(h + 1L):length(toks)])) {
      if (length(sub) == 1L) token_parent[[sub[1]]] <<- idx else build(sub, idx)
    }
    idx
  }
  if (window_size == 1L) { node_parent = 0L; token_parent = 1L } else build(seq_len(window_size), 0L)
  list(node_parent = as.integer(node_parent), token_parent = as.integer(token_parent),
       n_internal = length(node_parent))
}

BTreeCircuit = setRefClass("BTreeCircuit",
  contains = "ProbabilisticCircuit",
  fields = list(node_parent = "ANY", token_parent = "ANY", n_internal = "integer", n_trans = "integer"),
  methods = list(
    initialize = function(window_size = 8L, ranks = 32L) {
      # initializes the binary-tree circuit and its (fixed) tree topology
      callSuper(window_size = window_size, ranks = ranks)
      is_log_probs <<- TRUE
      topo = build_btree_topology(as.integer(window_size))
      node_parent  <<- topo$node_parent
      token_parent <<- topo$token_parent
      n_internal   <<- as.integer(topo$n_internal)
      n_trans      <<- as.integer(max(0L, topo$n_internal - 1L))
    },

    inject_head = function(model) {
      # injects root-prior, tree-transition and per-position emission layers; uniform mixtures at
      # every sum node (zeroed gates) + emissions copied from the STP lm_head -> BTree == FF at init
      ed = as.integer(model$embed_dim)
      vs = as.integer(model$vocab_size)
      layers = list(
        sum_unit_omega_init = nn$Linear(ed, ranks),                                        # root prior q(z_root)
        input_units_phi = nn$Linear(ed, as.integer(window_size * ranks * vs)),              # emissions [window, ranks, vocab]
        sum_unit_omega_transitions = nn$Linear(ed, as.integer(max(1L, n_trans) * ranks * ranks))  # tree edges q(z_child|z_parent)
      )
      model$heads$update(layers)

      # zero gates -> perfectly uniform mixture at every node of the tree
      nn$init$zeros_(model$heads[["sum_unit_omega_init"]]$weight)
      nn$init$zeros_(model$heads[["sum_unit_omega_init"]]$bias)
      nn$init$zeros_(model$heads[["sum_unit_omega_transitions"]]$weight)
      nn$init$zeros_(model$heads[["sum_unit_omega_transitions"]]$bias)

      with(torch$no_grad(), {
        if (!is.null(model$backbone$lm_head)) {
          stp_weight = model$backbone$lm_head$weight$detach()$clone()
        } else {
          stp_weight = model$backbone$get_output_embeddings()$weight$detach()$clone()
        }
        H = as.integer(stp_weight$shape[1])
        # [vocab, embed] -> [window, ranks, vocab, embed] (HMM layout) -> [window*ranks*vocab, embed]
        emission_init = stp_weight$unsqueeze(0L)$unsqueeze(0L)$
          expand(window_size, ranks, -1L, -1L)$
          reshape(-1L, H)
        model$heads[["input_units_phi"]]$weight$copy_(emission_init)
        nn$init$zeros_(model$heads[["input_units_phi"]]$bias)
      })
      invisible(model)
    },

    forward = function(model, hidden_states) {
      # per-position log-marginals via the binary tree, shape [batch, seq, window, vocab]
      batch_size = as.integer(hidden_states$shape[0])
      seq_len    = as.integer(hidden_states$shape[1])
      vocab_size = as.integer(model$vocab_size)

      log_init = F_$log_softmax(model$heads[["sum_unit_omega_init"]](hidden_states), dim = -1L)   # [B,S,r]
      flat_trans = model$heads[["sum_unit_omega_transitions"]](hidden_states)
      log_trans = F_$log_softmax(
        flat_trans$narrow(-1L, 0L, as.integer(n_trans * ranks * ranks))$view(batch_size, seq_len, n_trans, ranks, ranks),
        dim = -1L)                                                                                # [B,S,n_trans,r,r]
      flat_emiss = model$heads[["input_units_phi"]](hidden_states)
      log_emiss = F_$log_softmax(flat_emiss$view(batch_size, seq_len, window_size, ranks, vocab_size), dim = -1L)  # [B,S,W,r,V]

      log_p = vector("list", n_internal)
      log_p[[1]] = log_init
      if (n_internal >= 2L) for (k in 2:n_internal) {
        curr_trans = log_trans$select(2L, as.integer(k - 2L))                                     # [B,S,r_parent,r_child]
        log_p[[k]] = torch$logsumexp(log_p[[node_parent[k]]]$unsqueeze(-1L) + curr_trans, dim = 2L)  # [B,S,r_child]
      }
      log_marginal_probs = vector("list", window_size)
      for (i in seq_len(window_size)) {
        curr_emiss = log_emiss$select(2L, as.integer(i - 1L))                                     # [B,S,r,V]
        log_marginal_probs[[i]] = torch$logsumexp(log_p[[token_parent[i]]]$unsqueeze(-1L) + curr_emiss, dim = 2L)  # [B,S,V]
      }
      torch$stack(log_marginal_probs, dim = 2L)                                                   # [B,S,W,V]
    },

    get_draft_probs = function(model, embeddings) {
      # returns root prior, tree transitions and per-position emissions as probabilities
      with(torch$no_grad(), {
        last_emb = embeddings$select(1L, -1L)$unsqueeze(1L)                                        # [B,1,d]
        batch_size = as.integer(last_emb$shape[0])
        vocab_size = as.integer(model$vocab_size)

        init_probs = F_$softmax(model$heads[["sum_unit_omega_init"]](last_emb)$squeeze(1L), dim = -1L)  # [B,r]
        flat_trans = model$heads[["sum_unit_omega_transitions"]](last_emb)
        trans_probs = F_$softmax(
          flat_trans$narrow(-1L, 0L, as.integer(n_trans * ranks * ranks))$view(batch_size, n_trans, ranks, ranks),
          dim = -1L)                                                                              # [B,n_trans,r,r]
        flat_emiss = model$heads[["input_units_phi"]](last_emb)
        emiss_probs = F_$softmax(flat_emiss$view(batch_size, window_size, ranks, vocab_size), dim = -1L)  # [B,W,r,V]
        list(
          init  = as.array(init_probs$cpu()$numpy()),    # [batch, ranks]
          trans = as.array(trans_probs$cpu()$numpy()),   # [batch, n_trans, ranks, ranks]
          emiss = as.array(emiss_probs$cpu()$numpy())    # [batch, window, ranks, vocab]
        )
      })
    },

    get_full_vocab_dist = function(probs, step, batch_idx = 1L) {
      # per-position marginal q(x_step): propagate latent marginals root->leaf, then emit
      p = vector("list", n_internal)
      p[[1]] = probs$init[batch_idx, ]
      if (n_internal >= 2L) for (k in 2:n_internal) {
        p[[k]] = as.vector(p[[node_parent[k]]] %*% probs$trans[batch_idx, k - 1L, , ])
      }
      v = as.vector(p[[token_parent[step]]] %*% probs$emiss[batch_idx, step, , ])
      v / sum(v)
    },

    generate_draft = function(probs, batch_idx = 1L) {
      # marginal-argmax draft per window position (same convention as CP)
      sapply(seq_len(window_size), function(t) which.max(.self$get_full_vocab_dist(probs, t, batch_idx)) - 1L)
    },

    compute_prefix_probs = function(probs, draft_tokens, batch_idx = 1L) {
      # joint probability of each observed prefix via a bottom-up tree contraction
      ws = length(draft_tokens)
      q_values = numeric(ws)
      for (t in seq_len(ws)) {
        node_value = vector("list", n_internal)
        for (k in n_internal:1) {
          v = rep(1.0, ranks)
          for (j in which(node_parent == k)) v = v * as.vector(probs$trans[batch_idx, j - 1L, , ] %*% node_value[[j]])
          for (i in which(token_parent == k)) if (i <= t) v = v * probs$emiss[batch_idx, i, , draft_tokens[i] + 1L]
          node_value[[k]] = v
        }
        q_values[t] = sum(probs$init[batch_idx, ] * node_value[[1]])
      }
      q_values
    },

    get_conditional_dist = function(probs, accepted_prefix, batch_idx = 1L) {
      # distribution over the next token given the accepted prefix: tree contraction carrying the
      # vocab dimension along the query token's path (other tokens after the prefix marginalised out)
      s = length(accepted_prefix)
      query_pos = s + 1L
      node_value = vector("list", n_internal)
      has_vocab = logical(n_internal)
      for (k in n_internal:1) {
        v = rep(1.0, ranks); hv = FALSE
        for (j in which(node_parent == k)) {
          Tj = probs$trans[batch_idx, j - 1L, , ]
          if (has_vocab[j]) { v = v * (Tj %*% node_value[[j]]); hv = TRUE } else v = v * as.vector(Tj %*% node_value[[j]])
        }
        for (i in which(token_parent == k)) {
          if (i == query_pos) { v = v * probs$emiss[batch_idx, i, , ]; hv = TRUE }
          else if (i <= s) v = v * probs$emiss[batch_idx, i, , accepted_prefix[i] + 1L]
        }
        node_value[[k]] = v; has_vocab[k] = hv
      }
      res = as.vector(probs$init[batch_idx, ] %*% node_value[[1]])
      denom = sum(res)
      if (denom > 0) res / denom else res
    }
  )
)

create_circuit = function(type, window_size = 8L, ranks = 32L) {
  switch(tolower(type),
    "hmm"   = HMMCircuit$new(window_size = window_size, ranks = ranks),
    "ff"    = FFCircuit$new(window_size = window_size, ranks = ranks),
    "cp"    = CPCircuit$new(window_size = window_size, ranks = ranks),
    "btree" = BTreeCircuit$new(window_size = window_size, ranks = ranks),
    stop(sprintf("Tipo di circuito sconosciuto: '%s'. Usa 'hmm', 'ff', 'cp' o 'btree'.", type))
  )
}
