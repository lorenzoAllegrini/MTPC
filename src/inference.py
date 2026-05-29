import sys
sys.modules['torchvision'] = None

import torch
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader

from models.probabilistic_circuits import FF, CanonicPolyidiac, MTPC_HMM
from models.mtp_llm import MTP_LLM
from utils import MTPChatDataset, get_byt5_preprocess_function, load_tulu_dataset, CHAT_TEMPLATE


HEAD_CLASS = CanonicPolyidiac
MODEL_ID = "google/byt5-small"
WINDOW_SIZE = 6
MAX_LEN = 2048
NUM_VAL_SAMPLES = 200
LORA_PATH = "saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6"
WEIGHTS_PATH = "saved_models/mtp_head_cp_w6_final.pth"
TASK_FILTER = "ai2-adapt-dev/flan_v2_converted"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    model.heads.load_state_dict(state_dict)
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
            
            _, mtp_logits, hidden_states = model(input_ids, attention_mask=attention_mask)
            
            valid_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
            
            if len(valid_indices) == 0:
                continue
            
            test_idx = valid_indices[len(valid_indices) // 2].item()
            
            # Context up to test_idx (input for the head at test_idx)
            context_ids = input_ids[0, :test_idx + 1]
            
            # True next tokens (ground truth for window)
            true_ids = labels[0, test_idx + 1 : test_idx + 1 + WINDOW_SIZE]
            
            if head_name in ['CanonicPolyidiac', 'MTPC_HMM']:
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
