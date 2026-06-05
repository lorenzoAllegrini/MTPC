import sys
sys.modules['torchvision'] = None
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from models.probabilistic_circuits import CanonicPolyidiac
from models.mtp_llm import MTP_LLM
from utils import load_tulu_dataset, CHAT_TEMPLATE

def main():
    MODEL_ID = "google/byt5-small"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "/Users/lorenzoallegrini/Documents/MTP/saved_models"
    lora_path = os.path.join(save_dir, "lora_cp_w6", "mtp_backbone_lora_cp_w6")
    weights_path = os.path.join(save_dir, "mtp_head_cp_w6_final.pth")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load model
    model = MTP_LLM(MODEL_ID, CanonicPolyidiac, window_size=6, lora_path=lora_path)
    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("gate."):
            new_k = k.replace("gate.", "sum_unit_omega.", 1)
        elif k.startswith("emission_projs."):
            new_k = k.replace("emission_projs.", "input_units_phi.", 1)
        elif k.startswith("init_gate."):
            new_k = k.replace("init_gate.", "sum_unit_omega_init.", 1)
        elif k.startswith("emissions."):
            new_k = k.replace("emissions.", "input_units_phi.", 1)
        elif k.startswith("transitions."):
            new_k = k.replace("transitions.", "sum_unit_omega_transitions.", 1)
        new_state_dict[new_k] = v
        
    model.heads.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset sample
    train_data, _ = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=15000)
    
    # Find the correct sample index dynamically
    sample_idx = None
    for idx, item in enumerate(train_data):
        if "chopping coconuts" in item['messages'][0]['content']:
            sample_idx = idx
            break
            
    if sample_idx is None:
        print("Error: chopping coconuts sample not found in dataset!")
        return
        
    print(f"Using chopping coconuts sample at train index: {sample_idx}")
    messages = train_data[sample_idx]["messages"]
    
    pfx = "One can ch"
    
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=True
    )
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    initial_decoder_ids = tokenizer.encode(pfx, add_special_tokens=False, return_tensors="pt").to(device)
    
    print(f"Prompt length (P): {prompt_ids.shape[1]}")
    print(f"Prefix tokens: {initial_decoder_ids[0].tolist()}")
    
    P = prompt_ids.shape[1]
    decoder_ids_a = torch.zeros((1, P + 1), dtype=torch.long, device=device)
    decoder_ids_a = torch.cat([decoder_ids_a, initial_decoder_ids], dim=1)
    
    generated_tokens = initial_decoder_ids
    encoder_input_ids = torch.cat([prompt_ids, generated_tokens], dim=1)
    
    # Forward A
    with torch.no_grad():
        outputs_a = model.backbone(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_ids_a,
            use_cache=False,
            output_hidden_states=True
        )
    logits_a = outputs_a.logits
    next_token_probs_a = F.softmax(logits_a[0, -1, :], dim=-1)
    top_a = torch.topk(next_token_probs_a, 5)
    
    print("\n--- Predictions with Method A (Padded/Aligned positions) ---")
    for idx, (val, tok) in enumerate(zip(top_a.values, top_a.indices)):
        char = tokenizer.decode([tok.item()])
        print(f"  Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")

    # Get CP head predictions at the last position
    h_states = outputs_a.decoder_hidden_states[-1] # [1, L, embed_dim]
    last_emb = h_states[:, -1:, :] # [1, 1, embed_dim]
    
    # 1. Gate probs
    gate_logits = model.heads.sum_unit_omega(last_emb)
    gate_probs = F.softmax(gate_logits, dim=-1).squeeze(0).squeeze(0)
    top_gate = torch.topk(gate_probs, 5)
    print("\n--- CP Head Gate Probs Top 5 ---")
    for idx, (val, rank) in enumerate(zip(top_gate.values, top_gate.indices)):
        print(f"  Rank {rank.item() + 1}: p={val.item():.6f}")

    # 2. Marginal logits at each window step
    flat_input_units = model.heads.input_units_phi(last_emb)
    input_units = flat_input_units.view(1, 1, model._circuit.ranks, model._circuit.window_size, model.vocab_size).squeeze(1)
    
    # log_weights
    log_weights = F.log_softmax(gate_logits, dim=-1).squeeze(1) # [1, ranks]
    log_token_probs = F.log_softmax(input_units, dim=-1) # [1, ranks, window_size, vocab_size]
    
    # Log marginal probs: [1, window_size, vocab_size]
    log_weights_exp = log_weights.unsqueeze(-1).unsqueeze(-1)
    from models.probabilistic_circuits import stable_logsumexp
    log_marginal_probs = stable_logsumexp(log_weights_exp + log_token_probs, dim=1) # [1, window_size, vocab_size]
    marginal_probs = torch.exp(log_marginal_probs)[0] # [window_size, vocab_size]
    
    print("\n--- CP Head Marginal Predictions ---")
    for step in range(model._circuit.window_size):
        top_step = torch.topk(marginal_probs[step], 5)
        print(f"  Step {step + 1} Marginal Top 5:")
        for idx, (val, tok) in enumerate(zip(top_step.values, top_step.indices)):
            char = tokenizer.decode([tok.item()])
            print(f"    Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")
            
    # 3. Ancestral sample (generate_draft)
    print("\n--- CP Head Generated Drafts (Ancestral, 5 trials) ---")
    for trial in range(5):
        draft_tokens = model._circuit.generate_draft(h_states)[0]
        char = tokenizer.decode(draft_tokens.tolist())
        print(f"  Trial {trial+1}: {repr(char)} (Bytes: {draft_tokens.tolist()})")

    # 4. Marginal Argmax
    marginal_argmax = marginal_probs.argmax(dim=-1)
    char_marg = tokenizer.decode(marginal_argmax.tolist())
    print(f"\n--- CP Head Marginal Argmax: {repr(char_marg)} (Bytes: {marginal_argmax.tolist()}) ---")


if __name__ == "__main__":
    main()
