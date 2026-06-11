# searches for the easiest-to-predict prompt for the live demo (cp draft, argmax)
setwd("/Users/lorenzoallegrini/Documents/MTP/MTPC_project")
suppressMessages({
  source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
  source("mtpc/probabilistic_circuits.R"); source("mtpc/speculative_decoding.R")
})
CT = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained("google/byt5-small")
verifier = LLMWrapper(model_id = "google/byt5-small", lora_path = "saved_models/byt5_standard_lora_phase0", cheat = FALSE)
verifier$to(device); verifier$eval()
draft = LLMWrapper(model_id = "google/byt5-small", head_type = "cp", window_size = 6L, ranks = 32L,
                   lora_path = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6", cheat = FALSE)
draft$load_weights("saved_models/mtp_head_cp_w6_final.pth", device = "cpu")
draft$to(device); draft$eval()

# candidates from the benchmark dataset (same seed-42 selection)
splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
dataset = splits$test
set.seed(42)
sample_indices = sample(1:as.integer(dataset$num_rows), 50)
cands = list()
for (i in 1:10) {
  idx = sample_indices[i]
  msg = dataset[as.integer(idx - 1)]$messages
  p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CT, tokenize = FALSE, add_generation_prompt = TRUE)
  cands[[length(cands)+1]] = list(name = paste0("dataset_", i, "_idx", idx),
                                  prompt = p_txt,
                                  primer = substr(msg[[length(msg)]]$content, 1, 10))
}
# crafted easy candidates
mk = function(q) tokenizer$apply_chat_template(list(list(role="user", content=q)), chat_template = CT, tokenize = FALSE, add_generation_prompt = TRUE)
cands[[length(cands)+1]] = list(name="repeat_cat",  prompt=mk("Repeat exactly this sentence: the cat sat on the mat."), primer="the cat sa")
cands[[length(cands)+1]] = list(name="count_1_10",  prompt=mk("Count from 1 to 10, numbers separated by spaces."),      primer="1 2 3")
cands[[length(cands)+1]] = list(name="capital_fr",  prompt=mk("What is the capital of France?"),                        primer="The capita")
cands[[length(cands)+1]] = list(name="hello_5",     prompt=mk("Write the word hello five times."),                      primer="hello hell")

results = data.frame()
for (k in seq_along(cands)) {
  c0 = cands[[k]]
  pid = tokenizer$encode(c0$prompt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
  did = tokenizer$encode(c0$primer, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
  set.seed(7)
  r = generate_speculative(verifier_model = verifier, draft_model = draft, prompt_ids = pid,
        circuit = draft$circuit, tokenizer = tokenizer, initial_decoder_ids = did,
        max_new_tokens = 40L, verbose = FALSE, sampling = "argmax")
  acc = round(100 * r$total_accepted / r$total_proposed, 1)
  gen = gsub("\n", " ", safe_decode(tokenizer, as.integer(r$tokens)))
  results = rbind(results, data.frame(name = c0$name, acc = acc,
                  accepted = r$total_accepted, proposed = r$total_proposed,
                  text = substr(gen, 1, 70)))
  cat(sprintf("%-18s acc %5.1f%%  (%d/%d)  | %s\n", c0$name, acc, r$total_accepted, r$total_proposed, substr(gen, 1, 60)))
}
cat("\n=== ranking ===\n")
results = results[order(-results$acc), ]
print(results, row.names = FALSE)
saveRDS(cands, "/tmp/demo_cands.rds")
