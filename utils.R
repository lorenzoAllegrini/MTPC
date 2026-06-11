# shared non-analysis utility and helper functions for speculative decoding

CHAT_TEMPLATE = "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

# compute speculative decoding statistics from raw list of experiment results
compute_speculative_decoding_metrics = function(res_list) {
  # computes global and round-by-round speculative decoding metrics from experiment results

  n = length(res_list)

  # first go through the list once and add up the totals
  global_accepted = 0
  global_proposed = 0
  generated_texts = c()
  prompt_texts = c()
  for (i in 1:n) {
    x = res_list[[i]]
    global_accepted = global_accepted + x$total_accepted
    global_proposed = global_proposed + x$total_proposed
    generated_texts[i] = x$generated_text
    prompt_texts[i] = x$prompt_text
  }

  # find out the largest number of rounds so we know how wide the matrix is
  max_rounds = 0
  for (i in 1:n) {
    rounds = res_list[[i]]$round_accepted
    if (length(rounds) > max_rounds) {
      max_rounds = length(rounds)
    }
  }

  # build the matrix one row at a time, padding short rows with NA
  results_matrix = matrix(NA, nrow = n, ncol = max_rounds)
  for (i in 1:n) {
    rounds = res_list[[i]]$round_accepted
    for (j in 1:length(rounds)) {
      results_matrix[i, j] = rounds[j]
    }
  }

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
  filtered$select(seq(0L, min(max_samples, as.integer(filtered$num_rows)) - 1L))$train_test_split(test_size = 0.05, seed = 42L)
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
      pfx = paste0("<|", msg$role, "|>\n")
      sfx = "<|end|>\n"
      len = nchar(paste0(pfx, msg$content, sfx))
      if (msg$role != "assistant") {
        lbl[(off + 1L):min(off + len, length(ids))] = -100L
      } else {
        lbl[(off + 1L):min(off + nchar(pfx), length(ids))] = -100L
        lbl[min(off + len - nchar(sfx) + 1L, length(ids)):min(off + len, length(ids))] = -100L
      }
      off = off + len
      if (off >= length(ids)) break
    }
    if (all(lbl == -100L)) NULL else list(ids = ids, mask = as.integer(enc$attention_mask), lbl = lbl)
  })
  res = res[!sapply(res, is.null)]
  list(
    input_ids = lapply(res, function(x) x$ids),
    attention_mask = lapply(res, function(x) x$mask),
    labels = lapply(res, function(x) x$lbl)
  )
}

prepare_train_data = function(task_filter, tokenizer, max_len, batch_size, max_samples = 1000L) {
  # loads and preprocesses the training dataset and batches the inputs
  splits = load_tulu_dataset(task_filter, max_samples)
  create_batches(preprocess_conversations(splits$train$to_dict()$messages, tokenizer, max_len), batch_size)
}

