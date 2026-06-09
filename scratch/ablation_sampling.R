# ablation study: self-speculative decoding with argmax vs ancestral drafting,
# every circuit on the same 50 prompts (seed 42). saves benchmark_results/results_ablation_w6.rds
suppressMessages({
  source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
  source("mtpc/probabilistic_circuits.R"); source("mtpc/speculative_decoding.R")
})

MODEL_ID = "google/byt5-small"; WINDOW_SIZE = 6L; RANKS = 32L
N_SAMPLES = as.integer(Sys.getenv("ABL_N", "50"))
MAX_NEW_TOKENS = 60L
MODES = c("argmax", "ancestral")
CIRCUITS = c("ff", "cp", "hmm", "btree")
PATHS = list(
  ff    = list(lora = "saved_models/lora_ff_w6/mtp_backbone_lora_ff_w6",       head = "saved_models/mtp_head_ff_w6_final.pth"),
  cp    = list(lora = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6",       head = "saved_models/mtp_head_cp_w6_final.pth"),
  hmm   = list(lora = "saved_models/lora_hmm_w6/mtp_backbone_lora_hmm_w6",     head = "saved_models/mtp_head_hmm_w6_final.pth"),
  btree = list(lora = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", head = "saved_models/mtp_head_btree_w6_final.pth")
)
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
torch_py = import("torch"); gc_py = import("gc")
splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
dataset = splits$test

verifier = LLMWrapper(model_id = MODEL_ID, lora_path = "saved_models/byt5_standard_lora_phase0", cheat = FALSE)
verifier$to(device); verifier$eval()

run_one = function(draft, sampling) {
  set.seed(42)
  idx = sample(1:as.integer(dataset$num_rows), min(N_SAMPLES, as.integer(dataset$num_rows)))
  res_list = list(); t0 = Sys.time()
  for (i in seq_along(idx)) {
    msg = dataset[as.integer(idx[i] - 1)]$messages
    p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
    pfx = substr(msg[[length(msg)]]$content, 1, 10)
    pid = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    did = tokenizer$encode(pfx, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    r = generate_speculative(verifier_model = verifier, draft_model = draft, prompt_ids = pid,
          circuit = draft$circuit, tokenizer = tokenizer, initial_decoder_ids = did,
          max_new_tokens = MAX_NEW_TOKENS, verbose = FALSE, sampling = sampling)
    res_list[[i]] = list(round_accepted = r$round_accepted, total_accepted = r$total_accepted,
                         total_proposed = r$total_proposed, prompt_text = p_txt,
                         generated_text = safe_decode(tokenizer, as.integer(r$tokens)))
    rm(pid, did, r); gc(); gc_py$collect()
    tryCatch({ if (device$type == "mps") torch_py$mps$empty_cache() }, error = function(e) NULL)
  }
  m = compute_speculative_decoding_metrics(res_list)
  cat(sprintf("    [%s/%s] global=%.2f%%  (%.0fs)\n", "done", sampling,
              100 * m$global_accepted / m$global_proposed, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  list(acceptance_matrix = m$acceptance_matrix, generated_texts = m$generated_texts, prompt_texts = m$prompt_texts)
}

results = list()
for (ht in CIRCUITS) {
  cat(sprintf("\n==== circuit %s ====\n", toupper(ht)))
  draft = LLMWrapper(model_id = MODEL_ID, head_type = ht, window_size = WINDOW_SIZE, ranks = RANKS,
                     lora_path = PATHS[[ht]]$lora, cheat = FALSE)
  draft$load_weights(PATHS[[ht]]$head, device = "cpu"); draft$to(device); draft$eval()
  results[[ht]] = list()
  for (s in MODES) results[[ht]][[s]] = run_one(draft, s)
  rm(draft); gc(); gc_py$collect()
  saveRDS(results, "benchmark_results/results_ablation_w6.rds")   # incremental checkpoint
}
cat("\n[DONE] saved benchmark_results/results_ablation_w6.rds\n")
