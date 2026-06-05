source("mtpc/llm.R")
source("mtpc/utils.R")
source("utils.R")

MODEL_ID = "google/byt5-small"
VERIFIER_LORA_DIR = "saved_models/byt5-small-lora" # Let's find the correct path in the workspace
if (!file.exists(VERIFIER_LORA_DIR)) {
  VERIFIER_LORA_DIR = "saved_models/byt5_standard_lora_verifier/byt5_standard_lora"
}

device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
splits = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples = 1000L)
dataset = splits$test

verifier_model = LLMWrapper(model_id = MODEL_ID, lora_path = VERIFIER_LORA_DIR, cheat = TRUE)
verifier_model$to(device)
verifier_model$eval()

# Let's test on index 49 (which was Sample 1 in task-4884)
set.seed(42)
sample_indices = sample(1:as.integer(dataset$num_rows), 5)
idx = sample_indices[1] # corresponding to Sample 1

msg = dataset[as.integer(idx - 1)]$messages
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
p_txt = tokenizer$apply_chat_template(msg[1:(length(msg)-1)], chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)

prompt_ids = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)

# Standard autoregressive generation
generation_output = verifier_model$backbone$generate(
  input_ids = prompt_ids,
  max_new_tokens = 60L,
  do_sample = FALSE
)

generated_text = tokenizer$decode(as.integer(generation_output[0]$cpu()$numpy()))
cat(sprintf("\n[AUTOREGRESSIVE GENERATION]:\n'%s'\n", generated_text))
