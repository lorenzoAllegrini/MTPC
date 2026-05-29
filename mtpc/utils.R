get_device <- function() {
  if (torch$cuda$is_available()) {
    return(torch$device("cuda"))
  } else if (torch$backends$mps$is_available()) {
    return(torch$device("mps"))
  } else {
    return(torch$device("cpu"))
  }
}

pad_sequence <- function(seq_list, pad_value) {
  max_l <- max(sapply(seq_list, length))
  padded_list <- lapply(seq_list, function(s) {
    c(s, rep(pad_value, max_l - length(s)))
  })
  return(torch$tensor(padded_list, dtype = torch$long))
}

create_batches <- function(processed, batch_size, shuffle = TRUE) {
  all_input_ids <- processed$input_ids
  all_attention_mask <- processed$attention_mask
  all_labels <- processed$labels
  
  n <- length(all_input_ids)
  indices <- if (shuffle) sample(n) else seq_len(n)
  n_batches <- ceiling(n / batch_size)
  batches <- list()
  
  for (b in seq_len(n_batches)) {
    start <- (b - 1L) * batch_size + 1L
    end   <- min(b * batch_size, n)
    bi    <- indices[start:end]
    
    batches[[b]] <- list(
      input_ids      = pad_sequence(lapply(bi, function(i) all_input_ids[[i]]), 0L),
      attention_mask = pad_sequence(lapply(bi, function(i) all_attention_mask[[i]]), 0L),
      labels         = pad_sequence(lapply(bi, function(i) all_labels[[i]]), -100L)
    )
  }
  return(batches)
}

compute_mtpc_loss <- function(mtp_logits, labels, window_size, gamma = 0.9, is_log_probs = FALSE) {
  batch_size <- as.integer(mtp_logits$shape[0])
  seq_len    <- as.integer(mtp_logits$shape[1])
  vocab_size <- as.integer(mtp_logits$shape[as.integer(mtp_logits$dim() - 1L)])
  device     <- mtp_logits$device
  losses     <- list()
  
  for (j in seq_len(window_size)) {
    targets <- torch$roll(labels, shifts = as.integer(-j), dims = 1L)
    current_logits <- mtp_logits$select(2L, as.integer(j - 1L))
    flat_logits    <- current_logits$reshape(-1L, vocab_size)
    flat_labels    <- targets$reshape(-1L)
    
    if (is_log_probs) {
      step_loss <- F_$nll_loss(flat_logits, flat_labels, ignore_index = -100L, reduction = "mean")
    } else {
      step_loss <- F_$cross_entropy(flat_logits, flat_labels, ignore_index = -100L, reduction = "mean")
    }
    
    val <- step_loss$item()
    if (!is.nan(val) && !is.infinite(val)) {
      losses[[length(losses) + 1L]] <- (gamma^(j-1)) * step_loss
    }

  }
  
  if (length(losses) == 0L) return(torch$tensor(0.0, device = device))
  return(torch$stack(losses)$sum())
}

save_model <- function(model, head_type_name, window_size, save_dir = "saved_models") {
  filename <- sprintf("mtp_head_%s_w%d_final.pth", tolower(head_type_name), window_size)
  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  save_path <- file.path(save_dir, filename)
  model$save_weights(save_path)
  cat(sprintf("Model heads saved to %s\n", save_path))
}

load_model_weights <- function(model, weights_path, device) {
  model$load_weights(weights_path, device)
  invisible(model)
}
