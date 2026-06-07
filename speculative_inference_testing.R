source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")
source("mtpc/speculative_decoding.R")

# target models and architectures
MODEL_ID = "google/byt5-small"
PROBABILISTIC_HEADS = c("hmm", "ff", "cp", "btree")
WINDOW_SIZE = 6L
RANKS = 32L
MAX_LEN = 2048L
SHIFT_OFFSET_MINUS_1 = FALSE # set to TRUE only if loading legacy checkpoints trained with shifted target alignment
CHEAT = FALSE # keep FALSE: cheat feeds the full answer into byt5's encoder so it learns to copy instead of predict


N_SAMPLES = 100L

CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

run_inference_experiment = function(dataset, verifier_model, draft_model, circuit, tokenizer, n_samples = 100, max_new_tokens = 60L) {
  # runs speculative decoding experiments on a random sample of the dataset and reports metrics
  set.seed(42)
  sample_indices = sample(1:as.integer(dataset$num_rows), min(n_samples, as.integer(dataset$num_rows)))
  device = verifier_model$backbone$device
  
  # import python garbage collector
  gc_py = import("gc")
  
  res_list = list()
  for (i in seq_along(sample_indices)) {
    idx = sample_indices[i]
    cat(sprintf("\n--- Sample %d / %d (Dataset Index %d) ---\n", i, length(sample_indices), idx))
    
    # current chat
    msg = dataset[as.integer(idx - 1)]$messages
    p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
    # first characters of the message
    pfx = substr(msg[[length(msg)]]$content, 1, 10)
    
    prompt_ids = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    initial_decoder_ids = tokenizer$encode(pfx, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    
    res = generate_speculative(
      verifier_model = verifier_model, draft_model = draft_model, 
      prompt_ids = prompt_ids, 
      circuit = circuit, tokenizer = tokenizer, 
      initial_decoder_ids = initial_decoder_ids, 
      max_new_tokens = max_new_tokens, verbose = TRUE
    )
    
    res_list[[i]] = list(
      round_accepted = res$round_accepted, 
      total_accepted = res$total_accepted, 
      total_proposed = res$total_proposed, 
      prompt_text = p_txt, 
      generated_text = safe_decode(tokenizer, as.integer(res$tokens))
    )
    
    # explicitly clear r and python tensors/memory
    rm(prompt_ids, initial_decoder_ids, res)
    gc()
    gc_py$collect()
    
    # flush device-specific GPU/MPS cache if available
    tryCatch({
      if (device$type == "mps") {
        torch_py$mps$empty_cache()
      }
    }, error = function(e) {
      if (grepl("mps", as.character(device))) {
        torch_py$mps$empty_cache()
      }
    })
  }
  
  # compute metrics using the modular function
  metrics = compute_speculative_decoding_metrics(res_list)
  
  cat(sprintf("\nglobal acceptance: %.2f%%\n", (metrics$global_accepted / metrics$global_proposed) * 100))
  
  list(acceptance_matrix = metrics$acceptance_matrix, generated_texts = metrics$generated_texts, prompt_texts = metrics$prompt_texts)
}

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)

# ensure dataset is available
if (!exists("dataset")) {
  cat("\n[SYSTEM] Dataset 'dataset' not found in workspace. Loading and splitting 'ai2-adapt-dev/flan_v2_converted'...\n")
  splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
  dataset = splits$test
}

torch_py = import("torch")

# verifier model initialization
VERIFIER_LORA_DIR = "saved_models/byt5_standard_lora_phase0"
verifier_model = LLMWrapper(model_id = MODEL_ID, lora_path = VERIFIER_LORA_DIR, cheat = CHEAT)
verifier_model$to(device)
verifier_model$eval()

# accumulates per-head experiment results keyed by head_type
all_results = list()
get_inference_paths = function(head_type, window_size) {
  # explicit model paths (window 6 checkpoints); hmm uses its real adapter/head, not a .pth backbone
  return(switch(head_type,
    "ff"  = list(lora_dir = "saved_models/lora_ff_w6/mtp_backbone_lora_ff_w6", weights_path = "saved_models/mtp_head_ff_w6_final.pth"),
    "cp"  = list(lora_dir = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6",   weights_path = "saved_models/mtp_head_cp_w6_final.pth"),
    "hmm"   = list(lora_dir = "saved_models/lora_hmm_w6/mtp_backbone_lora_hmm_w6",     weights_path = "saved_models/mtp_head_hmm_w6_final.pth"),
    "btree" = list(lora_dir = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", weights_path = "saved_models/mtp_head_btree_w6_final.pth"),
    stop(sprintf("Unknown head_type: %s", head_type))
  ))

  # legacy auto-resolution (unreachable, kept for reference)
  # check if v1 reference models exist in saved_models
  v1_lora = sprintf("saved_models/lora_%s_w%d_phase1_v1", head_type, window_size)
  v1_weights = sprintf("saved_models/mtp_head_%s_w%d_phase1_v1.pth", head_type, window_size)
  
  if (file.exists(v1_weights) && file.exists(v1_lora)) {
    return(list(lora_dir = v1_lora, weights_path = v1_weights))
  }
  
  # check standard path in saved_models or legacy_models
  lora_dir = sprintf("saved_models/lora_%s_w%d/mtp_backbone_lora_%s_w%d", head_type, window_size, head_type, window_size)
  weights_path = sprintf("saved_models/mtp_head_%s_w%d_final.pth", head_type, window_size)
  
  if (!file.exists(weights_path) || !file.exists(lora_dir)) {
    # check legacy_models
    legacy_lora = sprintf("legacy_models/lora_%s_w%d/mtp_backbone_lora_%s_w%d", head_type, window_size, head_type, window_size)
    legacy_weights = sprintf("legacy_models/mtp_head_%s_w%d_final.pth", head_type, window_size)
    if (file.exists(legacy_weights) && file.exists(legacy_lora)) {
      lora_dir = legacy_lora
      weights_path = legacy_weights
    }
  }
  
  # fallbacks for specific custom paths in saved_models or legacy_models
  if (!file.exists(weights_path) || !file.exists(lora_dir)) {
    # check phase 1 in legacy_models
    alt_lora = sprintf("legacy_models/lora_%s_w%d_phase1", head_type, window_size)
    alt_weights = sprintf("legacy_models/mtp_head_%s_w%d_phase1.pth", head_type, window_size)
    if (file.exists(alt_weights) && file.exists(alt_lora)) {
      lora_dir = alt_lora
      weights_path = alt_weights
    }
  }
  
  return(list(lora_dir = lora_dir, weights_path = weights_path))
}

for (head_type in PROBABILISTIC_HEADS) {
  cat(sprintf("\nLoading draft model for: %s\n", head_type))
  paths = get_inference_paths(head_type, WINDOW_SIZE)
  
  draft_model = LLMWrapper(
    model_id = MODEL_ID, 
    head_type = head_type, 
    window_size = WINDOW_SIZE, 
    ranks = RANKS, 
    lora_path = paths$lora_dir,
    cheat = CHEAT
  )
   
  # load pretrained MTP head weights onto CPU before moving to target device
  if (file.exists(paths$weights_path)) {
    draft_model$load_weights(paths$weights_path, device = "cpu", shift_offset_minus_1 = SHIFT_OFFSET_MINUS_1)
  }
  draft_model$to(device)
  draft_model$eval()
  
  all_results[[head_type]] = run_inference_experiment(
    dataset = dataset,
    verifier_model = verifier_model,
    draft_model = draft_model,
    circuit = draft_model$circuit,
    tokenizer = tokenizer,
    n_samples = N_SAMPLES
  )
  rm(draft_model)
  gc()
}

# persist all head results to disk for later analysis
output_dir = "benchmark_results"
if (!dir.exists(output_dir)) dir.create(output_dir)
output_file = file.path(output_dir, sprintf("results_benchmark_w%d.rds", WINDOW_SIZE))
saveRDS(all_results, output_file)

