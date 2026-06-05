source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")
source("mtpc/probabilistic_circuits.R")

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

# Let's encode the prompt exactly as in inference
prompt_text = "An ice cream m"
prompt_ids = tokenizer$encode(prompt_text, add_special_tokens = FALSE, return_tensors = "pt")$to(device)

P = as.integer(prompt_ids$size(1L))
decoder_ids = torch$zeros(c(1L, P + 1L), dtype = torch$long, device = device)

hidden_states = draft_model$get_hidden_states(prompt_ids, decoder_ids)$x
probs = draft_model$circuit$get_draft_probs(draft_model, hidden_states)

# Let's run various draft generation methods

# 1. Current ancestral sampling (with multiple trials to see variation)
cat("\n--- 1. Current Ancestral Sampling (5 trials) ---\n")
for (trial in 1:5) {
  # Sample latent state from gate
  z = sample(seq_along(probs$gate[1, ]), 1, prob = probs$gate[1, ])
  draft = sapply(seq_len(window_size), function(t) {
    sample(seq_len(dim(probs$emiss)[4]) - 1L, 1, prob = probs$emiss[1, z, t, ])
  })
  cat(sprintf("  Trial %d: '%s' (Bytes: %s)\n", trial, tokenizer$decode(as.integer(draft)), paste(draft, collapse=", ")))
}

# 2. Marginal Argmax (FF-like argmax at each step)
cat("\n--- 2. Marginal Argmax ---\n")
marginal_argmax = sapply(seq_len(window_size), function(t) {
  dist = draft_model$circuit$get_full_vocab_dist(probs, t, 1L)
  which.max(dist) - 1L
})
cat(sprintf("  Draft: '%s' (Bytes: %s)\n", tokenizer$decode(as.integer(marginal_argmax)), paste(marginal_argmax, collapse=", ")))

# 3. Conditional Argmax (dynamically updating posterior)
cat("\n--- 3. Conditional Argmax ---\n")
conditional_argmax = integer(window_size)
for (t in seq_len(window_size)) {
  prefix = if (t > 1) conditional_argmax[1:(t-1)] else integer(0)
  dist = draft_model$circuit$get_conditional_dist(probs, prefix, 1L)
  conditional_argmax[t] = which.max(dist) - 1L
}
cat(sprintf("  Draft: '%s' (Bytes: %s)\n", tokenizer$decode(as.integer(conditional_argmax)), paste(conditional_argmax, collapse=", ")))

# 4. Marginal Sampling (5 trials)
cat("\n--- 4. Marginal Sampling (5 trials) ---\n")
for (trial in 1:5) {
  draft = sapply(seq_len(window_size), function(t) {
    dist = draft_model$circuit$get_full_vocab_dist(probs, t, 1L)
    sample(seq_len(length(dist)) - 1L, 1, prob = dist)
  })
  cat(sprintf("  Trial %d: '%s' (Bytes: %s)\n", trial, tokenizer$decode(as.integer(draft)), paste(draft, collapse=", ")))
}

# 5. Conditional Sampling (5 trials)
cat("\n--- 5. Conditional Sampling (5 trials) ---\n")
for (trial in 1:5) {
  draft = integer(window_size)
  for (t in seq_len(window_size)) {
    prefix = if (t > 1) draft[1:(t-1)] else integer(0)
    dist = draft_model$circuit$get_conditional_dist(probs, prefix, 1L)
    draft[t] = sample(seq_len(length(dist)) - 1L, 1, prob = dist)
  }
  cat(sprintf("  Trial %d: '%s' (Bytes: %s)\n", trial, tokenizer$decode(as.integer(draft)), paste(draft, collapse=", ")))
}
