# benchmarks the ff/cp/hmm drafts with self-speculative decoding and writes a log
source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")
source("mtpc/speculative_decoding.R")

# config
MODEL_ID            = "google/byt5-small"
PROBABILISTIC_HEADS = c("ff", "cp", "hmm")   # test all three new heads
WINDOW_SIZE         = 6L
RANKS               = 32L
MAX_LEN             = 2048L
N_SAMPLES           = 50L     # prompts per head (raise for tighter estimates; lower for a quick run)
MAX_NEW_TOKENS      = 60L
SHIFT_OFFSET_MINUS_1 = FALSE
CHEAT               = FALSE

# explicit paths to Marco's NEW models (avoids the v1 fallback picking old FF)
MODEL_PATHS = list(
  ff  = list(lora = "saved_models/lora_ff_w6_phase1",
             weights = "saved_models/mtp_head_ff_w6_phase1.pth"),
  cp  = list(lora = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6",
             weights = "saved_models/mtp_head_cp_w6_final.pth"),
  hmm = list(lora = "saved_models/lora_hmm_w6/mtp_backbone_lora_hmm_w6",
             weights = "saved_models/mtp_head_hmm_w6_final.pth")
)
VERIFIER_LORA_DIR = "saved_models/byt5_standard_lora_phase0_v1"

CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

# logging
dir.create("benchmark_results", showWarnings = FALSE)
LOG_PATH = file.path("benchmark_results", "benchmark_log.txt")
log_con  = file(LOG_PATH, open = "wt")
logln = function(...) {
  line = sprintf(...)
  cat(line, "\n", sep = "")
  cat(line, "\n", sep = "", file = log_con); flush(log_con)
}
logln("==================================================================")
logln(" MTPC SELF-SPECULATIVE DECODING BENCHMARK")
logln(" %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
logln(" model=%s  window n=%d  ranks=%d  samples/head=%d  max_new_tokens=%d",
      MODEL_ID, WINDOW_SIZE, RANKS, N_SAMPLES, MAX_NEW_TOKENS)
logln(" verifier=%s", VERIFIER_LORA_DIR)
logln("==================================================================")

# run logic (verbatim from speculative_inference_testing.R)
run_inference_experiment = function(dataset, verifier_model, draft_model, circuit, tokenizer, n_samples = 100, max_new_tokens = 60L) {
  set.seed(42)
  sample_indices = sample(1:as.integer(dataset$num_rows), min(n_samples, as.integer(dataset$num_rows)))
  device = verifier_model$backbone$device
  gc_py = import("gc")
  res_list = list()
  for (i in seq_along(sample_indices)) {
    idx = sample_indices[i]
    cat(sprintf("\n--- Sample %d / %d (Dataset Index %d) ---\n", i, length(sample_indices), idx))
    msg = dataset[as.integer(idx - 1)]$messages
    p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
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
    rm(prompt_ids, initial_decoder_ids, res); gc(); gc_py$collect()
    tryCatch({ if (device$type == "mps") torch_py$mps$empty_cache() },
             error = function(e) { if (grepl("mps", as.character(device))) torch_py$mps$empty_cache() })
  }
  metrics = compute_speculative_decoding_metrics(res_list)
  cat(sprintf("\nglobal acceptance: %.2f%%\n", (metrics$global_accepted / metrics$global_proposed) * 100))
  list(acceptance_matrix = metrics$acceptance_matrix, generated_texts = metrics$generated_texts, prompt_texts = metrics$prompt_texts)
}

# setup
device    = get_device()
logln(" device = %s", as.character(device))
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)

if (!exists("dataset")) {
  cat("\n[SYSTEM] Loading 'ai2-adapt-dev/flan_v2_converted' test split...\n")
  splits  = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
  dataset = splits$test
}

torch_py = import("torch")
verifier_model = LLMWrapper(model_id = MODEL_ID, lora_path = VERIFIER_LORA_DIR, cheat = CHEAT)
verifier_model$to(device); verifier_model$eval()

# benchmark loop
all_results = list()
summary_rows = list()
for (head_type in PROBABILISTIC_HEADS) {
  p = MODEL_PATHS[[head_type]]
  logln("")
  logln("------------------------------------------------------------------")
  logln(" [%s]  head=%s", toupper(head_type), basename(p$weights))
  logln("        lora=%s", p$lora)
  logln("------------------------------------------------------------------")

  draft_model = LLMWrapper(model_id = MODEL_ID, head_type = head_type,
                           window_size = WINDOW_SIZE, ranks = RANKS,
                           lora_path = p$lora, cheat = CHEAT)
  if (file.exists(p$weights)) {
    draft_model$load_weights(p$weights, device = "cpu", shift_offset_minus_1 = SHIFT_OFFSET_MINUS_1)
  } else {
    logln("   [WARNING] weights not found: %s", p$weights)
  }
  draft_model$to(device); draft_model$eval()

  res = run_inference_experiment(dataset, verifier_model, draft_model,
                                 draft_model$circuit, tokenizer,
                                 n_samples = N_SAMPLES, max_new_tokens = MAX_NEW_TOKENS)
  all_results[[head_type]] = res

  # per-head metrics for the log
  am   = res$acceptance_matrix
  vals = am[!is.na(am)]
  mu   = mean(vals)                       # mean accepted tokens per draft/verify round = mu_acc in [0, n]
  pct  = mu / WINDOW_SIZE * 100
  tot  = sum(vals); rounds = length(vals)
  dist = table(factor(vals, levels = 0:WINDOW_SIZE))
  logln("   prompts=%d  rounds=%d  tokens accepted=%d", N_SAMPLES, rounds, as.integer(tot))
  logln("   mean acceptance rate (mu_acc) = %.3f tokens/round   (= %.1f%% of the %d-token window)",
        mu, pct, WINDOW_SIZE)
  logln("   per-round accepted distribution (0..%d): %s", WINDOW_SIZE,
        paste(sprintf("%d:%d", 0:WINDOW_SIZE, as.integer(dist)), collapse = "  "))
  summary_rows[[head_type]] = c(mu_acc = mu, pct = pct, rounds = rounds)

  rm(draft_model); gc()
}

# save rds
output_file = file.path("benchmark_results", sprintf("results_benchmark_w%d.rds", WINDOW_SIZE))
saveRDS(all_results, output_file)
logln("")
logln(" saved raw results -> %s", output_file)

# final summary table
logln("")
logln("==================================================================")
logln(" SUMMARY  —  mean acceptance rate, window n=%d, %d prompts/head", WINDOW_SIZE, N_SAMPLES)
logln("------------------------------------------------------------------")
logln("   %-6s | %-22s | %-12s", "head", "mu_acc (tokens/round)", "acceptance %")
logln("   %s", strrep("-", 50))
for (h in PROBABILISTIC_HEADS) {
  r = summary_rows[[h]]
  if (!is.null(r)) logln("   %-6s | %-22.3f | %.1f %%", toupper(h), r["mu_acc"], r["pct"])
}
logln("==================================================================")
logln(" DONE. Full log: %s", LOG_PATH)
close(log_con)
cat("\n[OK] Benchmark complete. Log written to", LOG_PATH, "\n")
