get_device = function() {
  # gets the best available torch device (cuda, mps, or cpu)
  if (torch$cuda$is_available()) {
    return(torch$device("cuda"))
  } else if (torch$backends$mps$is_available()) {
    return(torch$device("mps"))
  } else {
    return(torch$device("cpu"))
  }
}

pad_sequence = function(seq_list, pad_value) {
  # pads a list of sequences to the maximum length and converts them to a torch tensor
  max_l = max(sapply(seq_list, length))
  padded_list = lapply(seq_list, function(s) {
    c(s, rep(pad_value, max_l - length(s)))
  })
  return(torch$tensor(padded_list, dtype = torch$long))
}

create_batches = function(processed, batch_size, shuffle = TRUE) {
  # groups processed inputs into padded batches for model training or inference
  n = length(processed$input_ids)
  idx = split(if (shuffle) sample(n) else seq_len(n), ceiling(seq_len(n) / batch_size))
  lapply(idx, function(bi) {
    list(
      input_ids      = pad_sequence(processed$input_ids[bi], 0L),
      attention_mask = pad_sequence(processed$attention_mask[bi], 0L),
      labels         = pad_sequence(processed$labels[bi], -100L)
    )
  })
}

compute_mtpc_loss = function(mtp_logits, labels, window_size, gamma = 0.8, is_log_probs = FALSE) {
  # computes the multi-token prediction loss weighted by an exponential decay factor
  
  # mtp_logits shape [batch_size, seq_len, window_size, vocab_size]; labels shape [batch_size, seq_len]
  vocab_size = as.integer(mtp_logits$shape[as.integer(mtp_logits$dim() - 1L)])
  loss = 0
  for (j in seq_len(window_size)) {
    # roll targets to align with future tokens, shape: [batch_size, seq_len]
    targets = torch$roll(labels, shifts = as.integer(-(j - 1L)), dims = 1L)
    # extract logits for step j-1, shape: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
    flat_logits = mtp_logits$select(2L, as.integer(j - 1L))$reshape(-1L, vocab_size)
    step_loss = if (is_log_probs) F_$nll_loss(flat_logits, targets$reshape(-1L), ignore_index = -100L) else F_$cross_entropy(flat_logits, targets$reshape(-1L), ignore_index = -100L)
    loss = loss + (gamma^(j-1)) * step_loss
  }
  return(loss)
}

save_model = function(model, head_type_name, window_size, save_dir = "saved_models") {
  # saves the weights of the model heads to a specified directory
  filename = sprintf("mtp_head_%s_w%d_final.pth", tolower(head_type_name), window_size)
  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  save_path = file.path(save_dir, filename)
  model$save_weights(save_path)
}

load_model_weights = function(model, weights_path, device) {
  # loads the weights of the model heads from a file path onto a specified device
  model$load_weights(weights_path, device)
  invisible(model)
}

safe_decode = function(tokenizer, tokens) {
  # decodes the tokens back into a string
  as.character(tokenizer$decode(as.list(as.integer(tokens))))
}
