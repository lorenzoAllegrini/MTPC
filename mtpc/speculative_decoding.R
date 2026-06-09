library(reticulate)

self_speculative_decoding_step = function(verifier_model, draft_model, prompt_ids, decoder_ids, circuit, lookahead_k = 4, draft_encoder_outputs = NULL, verifier_encoder_outputs = NULL, attention_mask = NULL, tokenizer = NULL, verbose = FALSE, sampling = "argmax") {
  # function that recreates the novel self speculative decoding introduced in the paper, pag 19

  hidden_states = llm_get_hidden_states(draft_model, prompt_ids, decoder_ids, attention_mask = attention_mask, encoder_outputs = draft_encoder_outputs)$x

  # draft the next tokens from the last hidden state
  probs = circuit$get_draft_probs(draft_model, hidden_states)
  drafted_tokens = circuit$generate_draft(probs, sampling = sampling)
  
  if (!is.null(tokenizer) && verbose) {draft_str = safe_decode(tokenizer, as.integer(drafted_tokens)); cat(" drafted tokens", draft_str, "\n")}

  # cumulative prefix probability for each token
  q_prefix = circuit$compute_prefix_probs(probs, drafted_tokens)
  
  verification = llm_verify_draft(
    model = verifier_model, 
    draft_tokens = drafted_tokens,
    encoder_outputs = verifier_encoder_outputs, 
    decoder_ids = decoder_ids,
    prompt_ids = prompt_ids,
    attention_mask = attention_mask
  )
  
  p_vals = verification$p
  next_p_dist = verification$next_p
  
  n = length(drafted_tokens)
  # each item t is the conditional probability of token t given all the previous tokens in the draft.
  q_cond = c(q_prefix[1], q_prefix[-1] / q_prefix[-n])

  # each token is accepted with probability p_vals/q_cond
  accepted_mask = runif(n) <= (p_vals / q_cond)

  # take the first false and set the remaining to false too
  first_false = which(!accepted_mask)
  s = if (length(first_false) > 0) first_false[1] - 1 else n
  
  if (verbose) {
    for (i in seq_len(n)) {
      token_str = if (!is.null(tokenizer)) safe_decode(tokenizer, as.integer(drafted_tokens[i])) else as.character(drafted_tokens[i])
      accepted = (i <= s)
      cat(" token", i, token_str, "Q", round(q_cond[i], 4), "P", round(p_vals[i], 4), if (accepted) "accepted" else "rejected", "\n")
    }
  }

  # generate the bonus token
  if (s < n) {
    p_dist_err = verification$full_p_dist$select(1L, as.integer(s))$squeeze(0L)
    
    # get the conditional distribution q(x_{s+1} | x_{1:s}) from the circuit
    accepted_draft = if (s > 0) drafted_tokens[1:s] else integer(0)
    q_dist_err = circuit$get_conditional_dist(probs, accepted_draft, batch_idx = 1L)
    
    q_dist_tensor = torch$tensor(q_dist_err, dtype = torch$float, device = p_dist_err$device)
    
    diff = p_dist_err - q_dist_tensor
    m_x = torch$clamp(diff, min = 0.0)
    Z = torch$sum(m_x)
    r_x = if (Z$item() > 0) m_x / Z else p_dist_err
    
    # cpu fallback for multinomial sampling due to PyTorch MPS backend bug
    next_token = torch$multinomial(r_x$cpu(), num_samples = 1L)$item()
    if (!is.null(tokenizer) && verbose) {
       cat(" bonus token", safe_decode(tokenizer, as.integer(next_token)), "\n")
    }
  } else {
    bonus_p_dist = verification$full_p_dist$select(1L, as.integer(n))$squeeze(0L)
    # cpu fallback for multinomial sampling due to PyTorch MPS backend bug
    next_token = torch$multinomial(bonus_p_dist$cpu(), num_samples = 1L)$item()
    if (!is.null(tokenizer) && verbose) {
       cat(" bonus token", safe_decode(tokenizer, as.integer(next_token)), "\n")
    }
  }
  
  accepted_draft = if (s > 0) drafted_tokens[1:s] else integer(0)
  final_new_tokens = c(accepted_draft, next_token)
  
  return(list(
    accepted_count = s,
    new_tokens = final_new_tokens
  ))
}


generate_speculative = function(verifier_model, draft_model, prompt_ids, circuit, tokenizer = NULL, initial_decoder_ids = NULL, max_new_tokens = 50L, eos_token_id = 1L, verbose = FALSE, sampling = "argmax") {
  # generates a sequence of tokens speculatively using a verifier model, a draft model, and a speculative circuit
  device = verifier_model$backbone$device
  
  with(torch$no_grad(), {

    # prompt_ids shape: [1, seq_len]
    if (!is.null(tokenizer) && verbose) {prompt_text = safe_decode(tokenizer, as.integer(prompt_ids$select(0L, 0L)$cpu()$numpy())); cat("prompt", prompt_text, "\n")}
    
    # mask padding tokens
    pad_id = as.integer(verifier_model$backbone$config$pad_token_id)
    attention_mask = prompt_ids$ne(pad_id)$to(torch$long)

    # in cheating mode skip precomputing encoder outputs since encoder inputs grow each step
    # otherwise precompute static prompt encoder outputs once to optimize inference speed
    if (verifier_model$cheat) {
      draft_encoder_outputs = NULL
      verifier_encoder_outputs = NULL
    } else {
      draft_encoder_outputs = draft_model$backbone$get_encoder()(prompt_ids, attention_mask = attention_mask)
      verifier_encoder_outputs = verifier_model$backbone$get_encoder()(prompt_ids, attention_mask = attention_mask)
    }
    
    # decoder starting token: prepend P + 1 zeros to match decoder_input_ids absolute position layout in relative position bias
    P = as.integer(prompt_ids$size(1L))
    decoder_ids = torch$zeros(c(1L, P + 1L), dtype = torch$long, device = device)
    
    # append initial prefix if provided
    if (!is.null(initial_decoder_ids)) { 
      # initial_decoder_ids shape: [decoder_seq_len] -> [1, decoder_seq_len]
      if (length(initial_decoder_ids$shape) == 1) initial_decoder_ids = initial_decoder_ids$unsqueeze(0L)
      decoder_ids = torch$cat(list(decoder_ids, initial_decoder_ids), dim = 1L)
    }
    
    initial_n = as.integer(decoder_ids$size(1L)) - (P + 1L)
    n_tokens_generated = initial_n
    round_idx = 1
    
    total_accepted = total_proposed = 0
    round_accepted = integer(0)
    
    while ((n_tokens_generated - initial_n) < max_new_tokens) {
      step_result = self_speculative_decoding_step(
        verifier_model  = verifier_model,
        draft_model = draft_model,
        prompt_ids = prompt_ids,
        decoder_ids = decoder_ids,
        circuit = circuit,
        lookahead_k = circuit$window_size,
        verifier_encoder_outputs = verifier_encoder_outputs,
        draft_encoder_outputs = draft_encoder_outputs,
        attention_mask = attention_mask,
        tokenizer = tokenizer,
        verbose = verbose,
        sampling = sampling
      )

      new_tokens = step_result$new_tokens
      # convert accepted tokens to tensor, new_tokens_tensor shape: [1, k + 1]
      new_tokens_tensor = torch$tensor(as.integer(new_tokens), dtype=torch$long, device=device)$view(c(1L, -1L))
      
      total_accepted = total_accepted + step_result$accepted_count
      total_proposed = total_proposed + circuit$window_size
      round_accepted = c(round_accepted, step_result$accepted_count)
      
      # decoder_ids shape: [1, seq_len + k + 1]
      decoder_ids = torch$cat(list(decoder_ids, new_tokens_tensor), dim=1L)
      n_tokens_generated = as.integer(decoder_ids$size(1L)) - (P + 1L)
      
      if (!is.null(tokenizer) && verbose) {
        full_toks = as.integer(decoder_ids[0]$cpu()$numpy())
        tokens_to_decode = if (length(full_toks) > P + 1L) full_toks[(P + 2L):length(full_toks)] else integer(0)
        current_text = safe_decode(tokenizer, tokens_to_decode)
        cat(" generated so far", current_text, "\n")
      }
      
      if (any(new_tokens == eos_token_id)) break
      if (!is.null(tokenizer)) {
        full_toks = as.integer(decoder_ids[0]$cpu()$numpy())
        tokens_to_decode = if (length(full_toks) > P + 1L) full_toks[(P + 2L):length(full_toks)] else integer(0)
        current_text = safe_decode(tokenizer, tokens_to_decode)
        if (grepl("<\\|end\\|>", current_text)) break
      }
      round_idx = round_idx + 1
    }
    
    list(
      tokens = as.integer(decoder_ids[0]$cpu()$numpy())[-(1:(P + 1L))],
      mean_acceptance = total_accepted / total_proposed,
      total_accepted = total_accepted,
      total_proposed = total_proposed,
      round_accepted = round_accepted
    )
  })
}
