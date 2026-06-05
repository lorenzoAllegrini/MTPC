source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")

CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

MODEL_ID = "google/byt5-small"
device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)

# Load CP draft model
head_type = "cp"
window_size = 6L
ranks = 32L

lora_dir = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6"
weights_path = "saved_models/mtp_head_cp_w6_final.pth"

draft_model = LLMWrapper(
  model_id = MODEL_ID, 
  head_type = head_type, 
  window_size = window_size, 
  ranks = ranks, 
  lora_path = lora_dir,
  cheat = TRUE
)
draft_model$load_weights(weights_path, device = "cpu", shift_offset_minus_1 = FALSE)
draft_model$to(device)
draft_model$eval()

# Let's encode the exact same chat-templated prompt
messages = list(
  list(role = "user", content = "Tell me a story about an ice cream man."),
  list(role = "assistant", content = "An ice cream m")
)
prompt_text = tokenizer$apply_chat_template(
  messages, chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = FALSE
)
# Strip ending <|end|>\n
prompt_text = paste0(strsplit(prompt_text, "An ice cream m")[[1]][1], "An ice cream m")

prompt_ids = tokenizer$encode(prompt_text, add_special_tokens = FALSE, return_tensors = "pt")$to(device)

P = as.integer(prompt_ids$size(1L))
decoder_ids = torch$zeros(c(1L, P + 1L), dtype = torch$long, device = device)

hidden_states = draft_model$get_hidden_states(prompt_ids, decoder_ids)$x
probs = draft_model$circuit$get_draft_probs(draft_model, hidden_states)

# Print gate top 5
gate_probs = probs$gate[1, ]
top_gate = order(gate_probs, decreasing = TRUE)[1:5]
cat("\n--- R Gate Probs Top 5 ---\n")
for (g in top_gate) {
  cat(sprintf("  Rank %d: p=%.6f\n", g, gate_probs[g]))
}

# Print emissions at step 1 for the top rank
top_rank = top_gate[1]
emiss_probs = probs$emiss[1, top_rank, 1, ]
top_emiss = order(emiss_probs, decreasing = TRUE)[1:5]
cat("\n--- R Emissions (Step 1, Top Rank) Top 5 ---\n")
for (e in top_emiss) {
  char = tokenizer$decode(as.integer(e - 1L))
  cat(sprintf("  Token %d ('%s'): p=%.6f\n", e - 1L, char, emiss_probs[e]))
}

# Print marginal predictions at each step
cat("\n--- R Marginal Predictions ---\n")
gate_probs = probs$gate[1, ]
for (t in seq_len(window_size)) {
  marginal_dist = numeric(as.integer(tokenizer$vocab_size))
  for (z in seq_len(ranks)) {
    marginal_dist = marginal_dist + gate_probs[z] * probs$emiss[1, z, t, ]
  }
  top_tokens = order(marginal_dist, decreasing = TRUE)[1:5]
  cat(sprintf("  Step %d Marginal Top 5:\n", t))
  for (tok in top_tokens) {
    char = tokenizer$decode(as.integer(tok - 1L))
    cat(sprintf("    Token %d ('%s'): p=%.6f\n", tok - 1L, char, marginal_dist[tok]))
  }
}


# Test the draft generation in R
draft_tokens = draft_model$circuit$generate_draft(probs, 1L)
draft_text = tokenizer$decode(as.integer(draft_tokens))
cat("\n--- R Generated Draft (Marginal Argmax) ---\n")
cat(sprintf("  Draft: '%s' (Bytes: %s)\n", draft_text, paste(draft_tokens, collapse=", ")))

