import sys
sys.modules['torchvision'] = None

import torch
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader

from models.probabilistic_circuits import FF, CanonicPolyidiac, MTPC_HMM, BTree
from models.mtp_llm import MTP_LLM
from utils import MTPChatDataset, get_byt5_preprocess_function, load_tulu_dataset, CHAT_TEMPLATE, get_model_paths_python


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="MTP Speculative Decoding Qualitative Inference")
    parser.add_argument("--head", type=str, choices=["cp", "hmm", "ff", "btree"], default="cp", help="Target head class (cp, hmm, ff, btree)")
    parser.add_argument("--window_size", type=int, default=4, help="Speculative window size")
    parser.add_argument("--model_id", type=str, default="google/byt5-small", help="Pretrained model ID")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of validation samples to test")
    parser.add_argument("--lora_path", type=str, default=None, help="Custom LoRA path (overrides resolved path)")
    parser.add_argument("--weights_path", type=str, default=None, help="Custom weights path (overrides resolved path)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Compute device (default: cuda).")
    args = parser.parse_args()
    
    MODEL_ID = args.model_id
    WINDOW_SIZE = args.window_size
    NUM_VAL_SAMPLES = args.num_samples
    MAX_LEN = 2048
    TASK_FILTER = "ai2-adapt-dev/flan_v2_converted"
    
    if args.head == "cp":
        HEAD_CLASS = CanonicPolyidiac
    elif args.head == "hmm":
        HEAD_CLASS = MTPC_HMM
    elif args.head == "ff":
        HEAD_CLASS = FF
    elif args.head == "btree":
        HEAD_CLASS = BTree

    save_dir = "saved_models"
    LORA_PATH, WEIGHTS_PATH = get_model_paths_python(args.head, args.window_size, save_dir)
            
    # Override resolved paths if user provided custom paths
    if args.lora_path is not None:
        LORA_PATH = args.lora_path
    if args.weights_path is not None:
        WEIGHTS_PATH = args.weights_path
        
    # Use the requested device, falling back gracefully if it is unavailable.
    if args.device == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[WARNING] --device cuda requested but CUDA is unavailable; using '{fallback}'.")
        args.device = fallback
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("[WARNING] --device mps requested but MPS is unavailable; using 'cpu'.")
        args.device = "cpu"
    device = torch.device(args.device)
    head_name = HEAD_CLASS.__name__
    print(f"Device: {device}")
    print(f"MTP Head: {head_name}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Model ID: {MODEL_ID}")
    
    # Determina l'architettura
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    
    model = MTP_LLM(
        model_id=MODEL_ID,
        head_class=HEAD_CLASS,
        window_size=WINDOW_SIZE,
        lora_path=LORA_PATH
    )

    
    # Load weights into model.heads (matching R's load_model_weights)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    
    # Remap old keys to new paper-aligned keys for backward compatibility
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
        elif k.startswith("emission_") and (".weight" in k or ".bias" in k):
            parts = k.split(".", 1)
            num = parts[0].split("_")[1]
            new_k = f"input_units_phi_{num}.{parts[1]}"
        new_state_dict[new_k] = v
        
    model.heads.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    if is_encoder_decoder:
        print("Architettura: Encoder-Decoder (Seq2Seq). Uso AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer.chat_template = CHAT_TEMPLATE
        preprocess_fn = get_byt5_preprocess_function(tokenizer, MAX_LEN, CHAT_TEMPLATE)
    else:
        print("Architettura: Decoder-Only (CausalLM). Uso codifica byte custom (EvaByte)...")
        tokenizer = None
        preprocess_fn = get_evabyte_preprocess_function(MAX_LEN)
    
    print("\n--- Loading validation data ---")
    _, val_data = load_tulu_dataset(TASK_FILTER, max_samples=10000)
    
    val_data = val_data.select(range(min(NUM_VAL_SAMPLES, len(val_data))))
    
    val_data = val_data.map(
        preprocess_fn,
        batched=True,
        num_proc=1,
        remove_columns=val_data.column_names,
        desc="Pre-tokenizing validation data"
    )
    
    val_dataset = MTPChatDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("\n" + "=" * 60)
    print(f"QUALITATIVE VALIDATION - Head {head_name}")
    print("=" * 60)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass to get logits and hidden states (passing labels to prevent leakage and align decoder)
            _, mtp_logits, hidden_states = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            valid_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                continue
            
            test_idx = valid_indices[len(valid_indices) // 2].item()
            
            # Context up to test_idx (input for the head at test_idx)
            context_ids = input_ids[0, :test_idx]
            
            # True next tokens (ground truth for window): step 1 predicts labels[test_idx]
            true_ids = labels[0, test_idx : test_idx + WINDOW_SIZE]
            
            if head_name in ['CanonicPolyidiac', 'MTPC_HMM', 'BTree']:
                # These expect hidden states [1, 1, D]
                test_emb = hidden_states[:, test_idx:test_idx+1, :]
                predicted_ids = model._circuit.generate_draft(test_emb)[0]
            else:
                # FF head: mtp_logits is [B, L, W, V]
                predicted_logits = mtp_logits[0, test_idx] # [W, V]
                predicted_ids = predicted_logits.argmax(dim=-1) # [W]

            # Filtriamo eventuali token di ignore (-100) per evitare crash del tokenizer
            true_ids = true_ids[true_ids != -100]
            predicted_ids = predicted_ids[predicted_ids != -100]

            context_text = tokenizer.decode(context_ids)
            true_text = tokenizer.decode(true_ids)
            predicted_text = tokenizer.decode(predicted_ids)
            
            print(f"\n--- EXAMPLE {i+1} ---")
            print(f"CONTEXT (last 80 bytes): ...{context_text[-80:]!r}")
            print(f"GROUND TRUTH (window {WINDOW_SIZE}): {true_text!r} (Bytes: {true_ids.tolist()})")
            print(f"PREDICTED    (window {WINDOW_SIZE}): {predicted_text!r} (Bytes: {predicted_ids.tolist()})")

    
    print(f"\n{'=' * 60}")
    print("Inference completed.")
