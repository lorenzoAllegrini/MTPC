import sys
sys.modules['torchvision'] = None
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from models.probabilistic_circuits import CanonicPolyidiac
from models.mtp_llm import MTP_LLM

def main():
    MODEL_ID = "google/byt5-small"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    save_dir = "/Users/lorenzoallegrini/Documents/MTP/saved_models"
    lora_path = os.path.join(save_dir, "lora_cp_w6", "mtp_backbone_lora_cp_w6")
    weights_path = os.path.join(save_dir, "mtp_head_cp_w6_final.pth")
    
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
    
    # Render prompt with chat template
    messages = [
        {"role": "user", "content": "Tell me a story about an ice cream man."},
        {"role": "assistant", "content": "An ice cream m"}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=False
    )
    # Strip the ending <|end|>\n
    prompt_text = prompt_text.split("An ice cream m")[0] + "An ice cream m"
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    
    P = prompt_ids.shape[1]
    decoder_ids = torch.zeros((1, P + 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        h_states = model.get_hidden_states(prompt_ids, decoder_ids)
        embeddings = h_states["x"]
        last_emb = embeddings[:, -1:, :]
        
        gate_logits = model.heads.sum_unit_omega(last_emb) # [batch, 1, ranks]
        gate_probs = F.softmax(gate_logits, dim=-1).squeeze(1) # [batch, ranks]
        
        flat_input_units = model.heads.input_units_phi(last_emb)
        input_units = flat_input_units.view(
            1, 1, model._circuit.ranks, model._circuit.window_size, model.vocab_size
        )
        # Squeeze sequence dimension
        input_units = input_units.squeeze(1) # [batch, ranks, window_size, vocab_size]
        emiss_probs = F.softmax(input_units, dim=-1)
        
    gate_probs_np = gate_probs[0].cpu().numpy()
    top_gate = np.argsort(gate_probs_np)[::-1][:5]
    print("\n--- Python Gate Probs Top 5 ---")
    for g in top_gate:
        # Note: python is 0-indexed, R is 1-indexed, so we add 1 to rank to compare with R
        print(f"  Rank {g+1}: p={gate_probs_np[g]:.6f}")
        
    top_rank = top_gate[0]
    emiss_probs_np = emiss_probs[0, top_rank, 0].cpu().numpy()
    top_emiss = np.argsort(emiss_probs_np)[::-1][:5]
    print("\n--- Python Emissions (Step 1, Top Rank) Top 5 ---")
    for e in top_emiss:
        char = tokenizer.decode([int(e)])
        print(f"  Token {e} ({repr(char)}): p={emiss_probs_np[e]:.6f}")

    # Test the draft generation in Python
    draft_tokens = model._circuit.generate_draft(embeddings)[0]
    draft_text = tokenizer.decode(draft_tokens.tolist())
    print("\n--- Python Generated Draft (Current Method) ---")
    print(f"  Draft: {repr(draft_text)} (Bytes: {draft_tokens.tolist()})")

if __name__ == "__main__":
    main()
