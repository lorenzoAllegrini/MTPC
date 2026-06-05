import sys
sys.modules['torchvision'] = None
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from models.probabilistic_circuits import CanonicPolyidiac
from models.mtp_llm import MTP_LLM
from utils import load_tulu_dataset, get_byt5_preprocess_function, CHAT_TEMPLATE, MTPChatDataset

def main():
    MODEL_ID = "google/byt5-small"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "/Users/lorenzoallegrini/Documents/MTP/saved_models"
    lora_path = os.path.join(save_dir, "lora_cp_w6", "mtp_backbone_lora_cp_w6")
    weights_path = os.path.join(save_dir, "mtp_head_cp_w6_final.pth")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    
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
    
    # Test sample
    prompt_text = (
        "<|user|>\nQUESTION: Premise: \"A man on a beach chopping coconuts with machete.\"\nHypothesis: \"A man is extracting coconut milk.\"\nIs the hypothesis entailed by the premise?\nOptions:\n- yes\n- it is not possible to tell\n- no\n\nLet's solve it slowly:\n<|assistant|>\n"
    )
    pfx = "One can ch"
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    initial_decoder_ids = tokenizer.encode(pfx, add_special_tokens=False, return_tensors="pt").to(device)
    
    print(f"Prompt length (P): {prompt_ids.shape[1]}")
    print(f"Prefix tokens: {initial_decoder_ids[0].tolist()}")
    
    # We compare two methods:
    # Method A (Padded/Shifted - matching how Seq2Seq SFT training aligns positions)
    # decoder_ids shape: [1, P + 1] of zeros, followed by prefix tokens
    P = prompt_ids.shape[1]
    decoder_ids_a = torch.zeros((1, P + 1), dtype=torch.long, device=device)
    decoder_ids_a = torch.cat([decoder_ids_a, initial_decoder_ids], dim=1)
    
    # Method B (Standard decoder prefix)
    # decoder_ids shape: [1, 1] (decoder_start_token_id) followed by prefix tokens
    decoder_start_token_id = model.backbone.config.decoder_start_token_id
    decoder_ids_b = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    decoder_ids_b = torch.cat([decoder_ids_b, initial_decoder_ids], dim=1)
    
    # In cheating mode: we concatenate generated tokens to the encoder
    # Prompt is 258 tokens. Initial prefix is 10 tokens.
    generated_tokens = initial_decoder_ids
    encoder_input_ids = torch.cat([prompt_ids, generated_tokens], dim=1)
    
    print(f"Encoder input shape: {encoder_input_ids.shape}")
    print(f"Decoder ids A (Padded) shape: {decoder_ids_a.shape}")
    print(f"Decoder ids B (Standard) shape: {decoder_ids_b.shape}")
    
    # Forward A
    outputs_a = model.backbone(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_ids_a,
        use_cache=False,
        output_hidden_states=True
    )
    logits_a = outputs_a.logits
    next_token_probs_a = F.softmax(logits_a[0, -1, :], dim=-1)
    top_a = torch.topk(next_token_probs_a, 5)
    
    # Forward B
    outputs_b = model.backbone(
        input_ids=encoder_input_ids,
        decoder_input_ids=decoder_ids_b,
        use_cache=False,
        output_hidden_states=True
    )
    logits_b = outputs_b.logits
    next_token_probs_b = F.softmax(logits_b[0, -1, :], dim=-1)
    top_b = torch.topk(next_token_probs_b, 5)
    
    print("\n--- Predictions with Method A (Padded/Aligned positions) ---")
    for idx, (val, tok) in enumerate(zip(top_a.values, top_a.indices)):
        char = tokenizer.decode([tok.item()])
        print(f"  Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")
        
    print("\n--- Predictions with Method B (Standard unpadded positions) ---")
    for idx, (val, tok) in enumerate(zip(top_b.values, top_b.indices)):
        char = tokenizer.decode([tok.item()])
        print(f"  Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")

if __name__ == "__main__":
    main()
