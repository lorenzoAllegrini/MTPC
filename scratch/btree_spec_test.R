source("mtpc/llm.R"); source("mtpc/utils.R"); source("utils.R")
source("mtpc/probabilistic_circuits.R"); source("mtpc/speculative_decoding.R")
MODEL_ID = "google/byt5-small"; device = get_device()
tokenizer = transformers$AutoTokenizer$from_pretrained(MODEL_ID)
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"

# verifier = phase0 (as in speculative_inference_testing.R); draft = the new BTree
verifier_model = LLMWrapper(model_id = MODEL_ID, lora_path = "saved_models/byt5_standard_lora_phase0", cheat = FALSE)
verifier_model$to(device); verifier_model$eval()

draft_model = LLMWrapper(model_id = MODEL_ID, head_type = "btree", window_size = 6L, ranks = 32L,
                         lora_path = "saved_models/lora_btree_w6/mtp_backbone_lora_btree_w6", cheat = FALSE)
draft_model$load_weights("saved_models/mtp_head_btree_w6_final.pth", device = "cpu", shift_offset_minus_1 = FALSE)
draft_model$to(device); draft_model$eval()
cat("[OK] BTree draft loaded; topology node_parent=", paste(draft_model$circuit$node_parent, collapse=","),
    " token_parent=", paste(draft_model$circuit$token_parent, collapse=","), "\n")

msgs = list(list(role="user", content="Is the hypothesis entailed by the premise? Explain briefly."))
p_txt = tokenizer$apply_chat_template(msgs, chat_template = CHAT_TEMPLATE, tokenize = FALSE, add_generation_prompt = TRUE)
prompt_ids = tokenizer$encode(p_txt, add_special_tokens = FALSE, return_tensors = "pt")$to(device)
initial_decoder_ids = tokenizer$encode("The answer", add_special_tokens = FALSE, return_tensors = "pt")$to(device)

res = generate_speculative(verifier_model = verifier_model, draft_model = draft_model,
                           prompt_ids = prompt_ids, circuit = draft_model$circuit, tokenizer = tokenizer,
                           initial_decoder_ids = initial_decoder_ids, max_new_tokens = 12L, verbose = TRUE)
cat("\n==== BTree speculative test OK ====\n")
cat("accepted/proposed:", res$total_accepted, "/", res$total_proposed, "\n")
cat("generated:", safe_decode(tokenizer, as.integer(res$tokens)), "\n")
