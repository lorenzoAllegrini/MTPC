source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
source("mtpc/probabilistic_circuits.R"); source("mtpc/speculative_decoding.R")

MODEL_ID = "google/byt5-small"; WINDOW_SIZE = 6L; RANKS = 32L; CHEAT = FALSE
N_SAMPLES = 6L; MAX_NEW = 40L
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
torch_py = import("torch")

splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
dataset = splits$test

# explicit (head -> lora_dir, weights) for the NEW models
heads = list(
  ff  = list(lora = "saved_models/lora_ff_w6_phase1",                   w = "saved_models/mtp_head_ff_w6_phase1.pth"),
  cp  = list(lora = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6",   w = "saved_models/mtp_head_cp_w6_final.pth"),
  hmm = list(lora = "saved_models/lora_hmm_w6/mtp_backbone_lora_hmm_w6", w = "saved_models/mtp_head_hmm_w6_final.pth")
)

set.seed(42)
sample_indices = sample(1:as.integer(dataset$num_rows), min(N_SAMPLES, as.integer(dataset$num_rows)))

flush_mps = function() tryCatch({ if (device$type == "mps") torch_py$mps$empty_cache() }, error=function(e) NULL)

report = list()
for (h in names(heads)) {
  cat(sprintf("\n================ HEAD: %s ================\n", toupper(h)))
  # Self-speculative: the SAME model is both draft (MTP head) and verifier (its backbone STP head).
  m = LLMWrapper(model_id = MODEL_ID, head_type = h, window_size = WINDOW_SIZE,
                 ranks = RANKS, lora_path = heads[[h]]$lora, cheat = CHEAT)
  m$load_weights(heads[[h]]$w, device = "cpu", shift_offset_minus_1 = FALSE)
  m$to(device); m$eval()

  tot_acc = 0L; tot_prop = 0L; n_rounds = 0L; texts = c()
  for (i in seq_along(sample_indices)) {
    idx = sample_indices[i]
    msg = dataset[as.integer(idx - 1)]$messages
    p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
    pfx = substr(msg[[length(msg)]]$content, 1, 10)
    prompt_ids = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    initial_decoder_ids = tokenizer$encode(pfx, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
    res = generate_speculative(verifier_model = m, draft_model = m,
                               prompt_ids = prompt_ids, circuit = m$circuit, tokenizer = tokenizer,
                               initial_decoder_ids = initial_decoder_ids, max_new_tokens = MAX_NEW, verbose = FALSE)
    tot_acc = tot_acc + res$total_accepted; tot_prop = tot_prop + res$total_proposed
    n_rounds = n_rounds + length(res$round_accepted)
    gen = safe_decode(tokenizer, as.integer(res$tokens))
    if (i <= 3) texts = c(texts, sprintf("  [pfx '%s'] -> '%s'", pfx, substr(gen, 1, 70)))
    cat(sprintf("  sample %d/%d: acc=%d/%d\n", i, length(sample_indices), res$total_accepted, res$total_proposed)); flush.console()
    rm(prompt_ids, initial_decoder_ids, res); gc(); flush_mps()
  }
  mu_acc = tot_acc / n_rounds
  report[[h]] = list(mu_acc = mu_acc, acc_pct = 100 * tot_acc / tot_prop, n_rounds = n_rounds, texts = texts)
  rm(m); gc(); flush_mps()
}

cat("\n\n############## REPORT (self-speculative) ##############\n")
cat(sprintf("verifier = each draft's own backbone STP | window=%d ranks=%d | %d samples, max_new=%d | cheat=%s\n\n",
            WINDOW_SIZE, RANKS, N_SAMPLES, MAX_NEW, CHEAT))
cat(sprintf("%-5s | %-20s | %-12s | %s\n", "head", "mu_acc (tok/round)", "acceptance%", "rounds"))
cat(paste(rep("-", 62), collapse=""), "\n")
for (h in names(report)) {
  r = report[[h]]
  cat(sprintf("%-5s | %-20.3f | %-12.2f | %d\n", toupper(h), r$mu_acc, r$acc_pct, r$n_rounds))
}
cat("\n--- sample generations ---\n")
for (h in names(report)) { cat(sprintf("\n[%s]\n", toupper(h))); for (t in report[[h]]$texts) cat(t, "\n") }
