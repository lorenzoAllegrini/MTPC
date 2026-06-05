import os
import sys
import torch
from torch.utils.data import DataLoader

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from models.probabilistic_circuits import CanonicPolyidiac, FF
from models.mtp_llm import MTP_LLM
from utils import (
    MTPChatDataset, 
    compute_mtpc_loss, 
    get_byt5_preprocess_function, 
    load_tulu_dataset, 
    CHAT_TEMPLATE
)
from training import swap_model_head, init_cp_from_ff
from transformers import AutoTokenizer

def test_init_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Set parameters
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    window_size = 6
    ranks = 32
    max_samples = 10
    max_len = 2048
    batch_size = 2
    model_id = "google/byt5-small"

    # 1. Locate reference files ending in _v1
    ff_head_path = os.path.join(models_dir, f"mtp_head_ff_w{window_size}_phase1_v1.pth")
    lora_path = os.path.join(models_dir, f"lora_ff_w{window_size}_phase1_v1")

    print(f"Loading FF head weights from: {ff_head_path}")
    print(f"Loading LoRA adapter from: {lora_path}")

    assert os.path.exists(ff_head_path), f"FF head path not found: {ff_head_path}"
    assert os.path.exists(lora_path), f"LoRA adapter path not found: {lora_path}"

    # 2. Setup tokenizer and dataset
    print("Setting up dataset...")
    train_data, _ = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    preprocess_fn = get_byt5_preprocess_function(tokenizer, max_len, tokenizer.chat_template)

    train_data = train_data.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train data"
    )

    train_chat_dataset = MTPChatDataset(train_data)
    train_loader = DataLoader(train_chat_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate model with LoRA phase 1 backbone
    print("Instantiating MTP_LLM model...")
    model = MTP_LLM(
        model_id=model_id,
        head_class=CanonicPolyidiac,
        window_size=window_size,
        ranks=ranks,
        lora_path=lora_path,
        cheat=True  # Same as validation / final training
    )
    model.to(device)

    # 4. Swap head to CP
    print("Swapping head to CanonicPolyidiac...")
    swap_model_head(model, CanonicPolyidiac, window_size, ranks, device)

    # 5. Transfer FF weights to CP
    print("Transferring FF weights to CP...")
    ff_state_dict = torch.load(ff_head_path, map_location=device)
    init_cp_from_ff(model, ff_state_dict, window_size, ranks, device)

    # 6. Evaluate initial loss
    print("Evaluating initial loss...")
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            _, mtp_logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.8, is_log_probs=True)
            
            total_loss += loss.item()
            num_batches += 1
            print(f"Batch {num_batches} loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    print(f"\nAverage initial CP loss: {avg_loss:.4f}")
    if avg_loss < 10.0:
        print("SUCCESS: Initial loss is below 10!")
    else:
        print("FAILURE: Initial loss is not below 10.")

if __name__ == "__main__":
    test_init_loss()
