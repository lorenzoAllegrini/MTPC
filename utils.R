# Shared non-analysis utility and helper functions for speculative decoding

CHAT_TEMPLATE = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

# Compute speculative decoding statistics from raw list of experiment results
compute_speculative_decoding_metrics = function(res_list) {
  # computes global and round-by-round speculative decoding metrics from experiment results
  # 1. Direct vector extraction of texts
  generated_texts = sapply(res_list, function(x) x$generated_text)
  prompt_texts    = sapply(res_list, function(x) x$prompt_text)
  
  # 2. Sum global speculative metrics
  global_accepted = sum(sapply(res_list, function(x) x$total_accepted))
  global_proposed = sum(sapply(res_list, function(x) x$total_proposed))
  
  # 3. Process and pad the ragged acceptance list
  acceptance_list = lapply(res_list, function(x) x$round_accepted)
  max_rounds      = max(sapply(acceptance_list, length))
  
  padded_list = lapply(acceptance_list, function(x) {
    c(x, rep(NA, max_rounds - length(x)))
  })
  
  # Row-wise binding to build the aligned matrix
  results_matrix = do.call(rbind, padded_list)
  
  return(list(
    acceptance_matrix = results_matrix,
    generated_texts   = generated_texts,
    prompt_texts      = prompt_texts,
    global_accepted   = global_accepted,
    global_proposed   = global_proposed
  ))
}

load_tulu_dataset = function(task_filter, max_samples = 1000L) {
  # loads the tulu-3-sft-mixture dataset, filters by task, and splits into train and test sets
  ds = datasets_lib$load_dataset("allenai/tulu-3-sft-mixture", split = "train")
  idx = which(ds$select_columns("source")$to_dict()[["source"]] == task_filter)
  if (length(idx) == 0) return(NULL)
  filtered = ds$select(as.integer(idx - 1L))$shuffle(seed = 42L)
  filtered$select(seq(0L, min(max_samples, as.integer(filtered$num_rows)) - 1L))$train_test_split(test_size = 0.05)
}

preprocess_conversations = function(messages_list, tokenizer, max_len = 4096L, template = CHAT_TEMPLATE) {
  # tokenizes conversation messages and masks non-assistant tokens for training
  res = lapply(messages_list, function(conv) {
    txt = tokenizer$apply_chat_template(conv, chat_template = template, tokenize = FALSE, add_generation_prompt = FALSE)
    enc = tokenizer(txt, add_special_tokens = FALSE, truncation = TRUE, max_length = max_len)
    ids = as.integer(enc$input_ids)
    lbl = ids
    # off tracks cumulative byte position so we can mask the correct label indices per message
    off = 0L
    for (msg in conv) {
      pfx = sprintf("<|%s|>\n", msg$role)
      sfx = "<|end|>\n"
      len = nchar(paste0(pfx, msg$content, sfx), type = "bytes")
      if (msg$role != "assistant") {
        lbl[(off + 1L):min(off + len, length(ids))] = -100L
      } else {
        lbl[(off + 1L):min(off + nchar(pfx, type = "bytes"), length(ids))] = -100L
        lbl[min(off + len - nchar(sfx, type = "bytes") + 1L, length(ids)):min(off + len, length(ids))] = -100L
      }
      off = off + len
      if (off >= length(ids)) break
    }
    if (all(lbl == -100L)) NULL else list(ids = ids, mask = as.integer(enc$attention_mask), lbl = lbl)
  })
  res = res[!sapply(res, is.null)]
  list(
    input_ids = lapply(res, `[[`, "ids"),
    attention_mask = lapply(res, `[[`, "mask"),
    labels = lapply(res, `[[`, "lbl")
  )
}

prepare_train_data = function(task_filter, tokenizer, max_len, batch_size, max_samples = 1000L) {
  # loads and preprocesses the training dataset and batches the inputs
  splits = load_tulu_dataset(task_filter, max_samples)
  create_batches(preprocess_conversations(splits$train$to_dict()$messages, tokenizer, max_len), batch_size)
}

init_probabilistic_circuit = function(head_type, window_size, ranks) {
  # initializes a probabilistic circuit of the specified type
  switch(tolower(head_type),
    "ff" = FFCircuit$new(window_size = window_size, ranks = ranks),
    "hmm" = HMMCircuit$new(window_size = window_size, ranks = ranks),
    "cp" = CPCircuit$new(window_size = window_size, ranks = ranks),
    stop(sprintf("Tipo di testa non supportato: %s", head_type))
  )
}

get_model_paths = function(head_type, window_size) {
  # retrieves the lora save directory and head weights paths for a given model type
  t = tolower(head_type)
  if (t == "hmm") {
    list(
      lora_dir = sprintf("saved_models/lora_hmm_w%d/mtp_backbone_lora_mtpc_hmm_w%d_ft", window_size, window_size),
      weights_path = sprintf("saved_models/mtp_head_mtpc_hmm_w%d_ft.pth", window_size)
    )
  } else {
    list(
      lora_dir = sprintf("saved_models/lora_%s_w%d/mtp_backbone_lora_%s_w%d", t, window_size, t, window_size),
      weights_path = sprintf("saved_models/mtp_head_%s_w%d_final.pth", t, window_size)
    )
  }
}
