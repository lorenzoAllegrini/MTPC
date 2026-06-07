# benchmarks the newly trained btree draft on the same 50 prompts as the prior run,
# then merges the result into benchmark_results/results_benchmark_w6.rds
suppressMessages({
  source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
  source("mtpc/probabilistic_circuits.R"); source("mtpc/speculative_decoding.R")
})

MODEL_ID = "google/byt5-small"
WINDOW_SIZE = 6L; RANKS = 32L
N_SAMPLES = as.integer(Sys.getenv("BT_N", "50"))
MAX_NEW_TOKENS = 60L
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
torch_py = import("torch")

splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
dataset = splits$test

verifier_model = LLMWrapper(model_id = MODEL_ID, lora_path = "saved_models/byt5_standard_lora_phase0", cheat = FALSE)
verifier_model$to(device); verifier_model$eval()

draft_model = LLMWrapper(model_id = MODEL_ID, head_type = "btree", window_size = WINDOW_SIZE, ranks = RANKS,
                         lora_path = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", cheat = FALSE)
draft_model$load_weights("saved_models/mtp_head_btree_w6_final.pth", device = "cpu", shift_offset_minus_1 = FALSE)
draft_model$to(device); draft_model$eval()

set.seed(42)
sample_indices = sample(1:as.integer(dataset$num_rows), min(N_SAMPLES, as.integer(dataset$num_rows)))
gc_py = import("gc")
res_list = list()
t0 = Sys.time()
for (i in seq_along(sample_indices)) {
  idx = sample_indices[i]
  msg = dataset[as.integer(idx - 1)]$messages
  p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
  pfx = substr(msg[[length(msg)]]$content, 1, 10)
  prompt_ids = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
  initial_decoder_ids = tokenizer$encode(pfx, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
  res = generate_speculative(verifier_model = verifier_model, draft_model = draft_model,
        prompt_ids = prompt_ids, circuit = draft_model$circuit, tokenizer = tokenizer,
        initial_decoder_ids = initial_decoder_ids, max_new_tokens = MAX_NEW_TOKENS, verbose = FALSE)
  res_list[[i]] = list(round_accepted = res$round_accepted, total_accepted = res$total_accepted,
                       total_proposed = res$total_proposed, prompt_text = p_txt,
                       generated_text = safe_decode(tokenizer, as.integer(res$tokens)))
  el = as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("[%2d/%2d] acc %d/%d  | %.1fs elapsed (%.1fs/sample)\n",
              i, length(sample_indices), res$total_accepted, res$total_proposed, el, el / i))
  rm(prompt_ids, initial_decoder_ids, res); gc(); gc_py$collect()
  tryCatch({ if (device$type == "mps") torch_py$mps$empty_cache() }, error = function(e) NULL)
}

metrics = compute_speculative_decoding_metrics(res_list)
btree_res = list(acceptance_matrix = metrics$acceptance_matrix,
                 generated_texts = metrics$generated_texts, prompt_texts = metrics$prompt_texts)
saveRDS(btree_res, "benchmark_results/btree_new_result.rds")
cat(sprintf("\nNEW BTREE global acceptance: %.2f%%\n", (metrics$global_accepted / metrics$global_proposed) * 100))

# merge into the combined rds (keep hmm/ff/cp, replace btree) only on a full run
if (N_SAMPLES >= 50L) {
  all_results = readRDS("benchmark_results/results_benchmark_w6.rds")
  all_results[["btree"]] = btree_res
  saveRDS(all_results, "benchmark_results/results_benchmark_w6.rds")
  cat("[merged new btree into results_benchmark_w6.rds]\n")
}
