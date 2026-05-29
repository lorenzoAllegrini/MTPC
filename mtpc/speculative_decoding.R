library(reticulate)

self_speculative_decoding_step <- function(verifier_model, draft_model, prompt_ids, decoder_ids, circuit, lookahead_k = 4, draft_encoder_outputs = NULL, verifier_encoder_outputs = NULL, tokenizer = NULL, verbose = FALSE) {
  # 1. Otteniamo gli hidden states
  # Passiamo attention_mask per evitare il rumore del padding nell'encoder
  pad_id <- as.integer(verifier_model$backbone$config$pad_token_id)
  attention_mask <- prompt_ids$ne(pad_id)$to(torch$long)
  
  res_h <- engine_get_hidden_states(draft_model, prompt_ids, decoder_ids, attention_mask = attention_mask, encoder_outputs = draft_encoder_outputs)
  hidden_states <- res_h$x
  L_hs <- as.integer(hidden_states$size(1L))

  # 2. Generazione della proposta (Drafting)
  # Torniamo a usare h_{N-1} perché il log ha confermato che h_N predice "troppo avanti"
  probs <- circuit$get_draft_probs(draft_model, hidden_states$narrow(1L, 0L, as.integer(L_hs - 1L)))
  drafted_tokens <- circuit$generate_draft(probs) 
  
  if (!is.null(tokenizer) && verbose) {
    draft_str <- tokenizer$decode(as.integer(drafted_tokens))
    cat(sprintf("  [DRAFT] Proposta: '%s'\n", draft_str))
  }

  # 3. Probabilità del prefisso (Q) e verifica (P)
  q_prefix <- circuit$compute_prefix_probs(probs, drafted_tokens)
  
  verification <- engine_verify_draft(
    model           = verifier_model, 
    draft_tokens    = drafted_tokens,
    encoder_outputs = verifier_encoder_outputs, 
    decoder_ids     = decoder_ids,
    prompt_ids      = prompt_ids,
    attention_mask  = attention_mask
  )
  
  p_vals <- verification$p        
  next_p_dist <- verification$next_p
  
  n <- length(drafted_tokens)
  s <- 0
  
  if (verbose) cat("  [LOG] Verifica Speculativa:\n")
  for (i in seq_len(n)) {
    q_cond <- if (i == 1) q_prefix[1] else q_prefix[i] / q_prefix[i-1]
    p_val  <- p_vals[i]
    
    alpha <- runif(1) 
    accepted <- alpha <= p_val / q_cond
    
    if (verbose) {
      token_str <- if (!is.null(tokenizer)) tokenizer$decode(as.integer(drafted_tokens[i])) else as.character(drafted_tokens[i])
      cat(sprintf("    - [%d] '%s': Q=%.4f | P=%.4f -> %s\n", 
                  i, token_str, q_cond, p_val, 
                  if (accepted) "ACCETTATO" else "RIGETTATO"))
    }
    
    if (!accepted) {
      break 
    }
    s <- s + 1
  }
  
  # 4. Generazione del Bonus Token
  if (s < n) {
    p_dist_err <- verification$full_p_dist$select(1L, as.integer(s))$squeeze(0L)
    q_dist_err <- circuit$get_full_vocab_dist(probs, step = as.integer(s + 1L))
    q_dist_tensor <- torch$tensor(q_dist_err, dtype = torch$float, device = p_dist_err$device)
    
    diff <- p_dist_err - q_dist_tensor
    m_x <- torch$clamp(diff, min = 0.0)
    Z <- torch$sum(m_x)
    r_x <- m_x / Z
    
    next_token <- torch$multinomial(r_x, num_samples = 1L)$item()
    if (!is.null(tokenizer) && verbose) {
       cat(sprintf("    - [%d] Bonus (M): '%s' (Sostituzione)\n", s + 1, tokenizer$decode(as.integer(next_token))))
    }
  } else {
    bonus_p_dist <- verification$full_p_dist$select(1L, as.integer(n))$squeeze(0L)
    next_token <- torch$multinomial(bonus_p_dist, num_samples = 1L)$item()
    if (!is.null(tokenizer) && verbose) {
       cat(sprintf("    - [%d] Bonus (P): '%s'\n", n + 1, tokenizer$decode(as.integer(next_token))))
    }
  }
  
  accepted_draft <- if (s > 0) drafted_tokens[1:s] else integer(0)
  final_new_tokens <- c(accepted_draft, next_token)
  
  return(list(
    accepted_count = s,
    new_tokens = final_new_tokens
  ))
}

generate_speculative <- function(verifier_model, draft_model, prompt_ids, circuit, tokenizer = NULL, initial_decoder_ids = NULL, max_new_tokens = 50L, eos_token_id = 1L, verbose = FALSE) {
  device <- verifier_model$backbone$device
  
  reticulate::with(torch$no_grad(), {
    if (!is.null(tokenizer) && verbose) {
      # Usiamo select(0L, 0L) per estrarre la prima riga del batch in modo sicuro
      prompt_text <- tokenizer$decode(as.integer(prompt_ids$select(0L, 0L)$cpu()$numpy()))
      cat(sprintf("[SYSTEM] Prompt: '%s'\n", prompt_text))
    }
    
    pad_id <- as.integer(verifier_model$backbone$config$pad_token_id)
    attention_mask <- prompt_ids$ne(pad_id)$to(torch$long)
    
    draft_encoder_outputs <- draft_model$backbone$get_encoder()(prompt_ids, attention_mask = attention_mask)
    verifier_encoder_outputs <- verifier_model$backbone$get_encoder()(prompt_ids, attention_mask = attention_mask)
    
    decoder_start_token_id <- as.integer(verifier_model$backbone$config$decoder_start_token_id)
    decoder_ids <- torch$tensor(matrix(c(decoder_start_token_id), nrow = 1), dtype = torch$long, device = device)
    
    if (!is.null(initial_decoder_ids)) {
      if (length(initial_decoder_ids$shape) == 1) initial_decoder_ids <- initial_decoder_ids$unsqueeze(0L)
      decoder_ids <- torch$cat(list(decoder_ids, initial_decoder_ids), dim = 1L)
    }
    
    initial_n <- as.integer(decoder_ids$size(1L)) - 1
    n_tokens_generated <- initial_n
    round_idx <- 1
    
    total_accepted <- 0
    total_proposed <- 0
    round_accepted <- integer(0)
    
    while ((n_tokens_generated - initial_n) < max_new_tokens) {
      if (verbose) {
        current_text <- tokenizer$decode(as.integer(decoder_ids[0]$cpu()$numpy()))
        cat(sprintf("\n  --- ROUND %d (Tokens: %d) ---\n", round_idx, n_tokens_generated))
        cat(sprintf("  [CONTEXT] '%s'\n", current_text))
      }
      
      step_result <- self_speculative_decoding_step(
        verifier_model  = verifier_model,
        draft_model     = draft_model,
        prompt_ids      = prompt_ids,
        decoder_ids     = decoder_ids,
        circuit         = circuit,
        verifier_encoder_outputs = verifier_encoder_outputs,
        draft_encoder_outputs = draft_encoder_outputs,
        tokenizer       = tokenizer,
        verbose         = verbose
      )

      new_tokens <- step_result$new_tokens
      new_tokens_tensor <- torch$tensor(as.integer(new_tokens), dtype=torch$long, device=device)$view(c(1L, -1L))
      
      # Aggiornamento statistiche
      total_accepted <- total_accepted + step_result$accepted_count
      total_proposed <- total_proposed + circuit$window_size
      round_accepted <- c(round_accepted, step_result$accepted_count)
      
      decoder_ids <- torch$cat(list(decoder_ids, new_tokens_tensor), dim=1L)
      n_tokens_generated <- as.integer(decoder_ids$size(1L)) - 1
      
      if (any(new_tokens == eos_token_id)) break
      round_idx <- round_idx + 1
    }
    
    list(
      tokens = as.integer(decoder_ids[0]$cpu()$numpy())[-1],
      mean_acceptance = total_accepted / total_proposed,
      total_accepted = total_accepted,
      total_proposed = total_proposed,
      round_accepted = round_accepted
    )
  })
}
