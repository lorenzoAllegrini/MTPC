import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Import utility functions from the src/utils.py
from utils import (
    load_tulu_dataset, 
    get_byt5_preprocess_function, 
    MTPChatDataset, 
    CHAT_TEMPLATE
)

print("\n" + "="*50)
print("INIZIO FINE-TUNING ByT5 STANDARD (Senza MTPC)")
print("="*50)

model_id = "google/byt5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
best_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
max_len = 2048

# 1. Caricamento e Tokenizzazione Dataset
train_data, val_data = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=10000)


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.chat_template = CHAT_TEMPLATE
preprocess_fn = get_byt5_preprocess_function(tokenizer, max_len, tokenizer.chat_template)

train_data = train_data.map(preprocess_fn, batched=True, num_proc=6, remove_columns=train_data.column_names)
val_data = val_data.map(preprocess_fn, batched=True, num_proc=6, remove_columns=val_data.column_names)

train_loader = DataLoader(MTPChatDataset(train_data), batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(MTPChatDataset(val_data), batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

# 2. Setup Modello Standard con LoRA
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=best_dtype)
lora_config = LoraConfig(r=8, lora_alpha=16, bias="none", task_type="SEQ_2_SEQ_LM")
model_standard = get_peft_model(base_model, lora_config)
model_standard.to(device)

optimizer = Adam(model_standard.parameters(), lr=1e-3)

# 3. Training Loop Standard
model_standard.train()
for epoch in range(1):
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Standard Epoch {epoch+1} (Train)")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Per i modelli HuggingFace, passare le labels fa calcolare la cross-entropy loss internamente
        outputs = model_standard(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/(batch_idx+1):.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # Validation Phase Standard
    model_standard.eval()
    total_val_loss = 0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Standard Epoch {epoch+1} (Val)")
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model_standard(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss

            total_val_loss += val_loss.item()
            val_pbar.set_postfix(val_loss=f"{val_loss.item():.4f}")

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Standard Epoch {epoch+1} completata. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    model_standard.train()

# 4. Salvataggio
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "byt5_standard_lora_verifier")
model_standard.save_pretrained(save_path)
print(f"Modello standard salvato con successo in {save_path}")

