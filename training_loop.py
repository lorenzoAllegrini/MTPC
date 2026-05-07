import torch 
import numpy as np 
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim import Adam


class MTPChatDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length=512):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        conversation = self.raw_data[idx]['messages']
        
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', add_special_tokens=False, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        current_offset = 0
        for msg in conversation:
            msg_text = f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
            msg_len = len(msg_text.encode('utf-8'))
            
            if msg['role'] != "assistant":
                start = current_offset
                end = min(current_offset + msg_len, self.max_length)
                if start < self.max_length:
                    labels[start:end] = -100
            
            current_offset += msg_len
            if current_offset >= self.max_length: break

        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.9, ignore_index=-100):
    """
    Calcola la loss per Multi-Token Prediction con finestre sovrapposte[cite: 265].
    
    Args:
        mtp_logits: [B, L, n, V] - Logit prodotti dalla testa MTP.
        labels: [B, L] - ID dei token reali (con ignore_index per l'utente).
        window_size: n - Ampiezza della finestra di predizione.
        gamma: Fattore di sconto esponenziale[cite: 271].
    """
    combined_loss = 0.0
    
    for j in range(1, window_size + 1):
        current_logits = mtp_logits[:, :-j, j-1, :] 
        current_labels = labels[:, j:]
        step_loss = F.cross_entropy(
            current_logits.reshape(-1, current_logits.size(-1)),
            current_labels.reshape(-1),
            ignore_index=ignore_index
        )
        
        combined_loss += (gamma ** (j - 1)) * step_loss
        
    return combined_loss

from tqdm import tqdm

def train_mtp(model, train_loader, device, window_size=8, gamma=0.9):
    model.to(device)
    
    # Setup secondo il paper: addestramento della testa con backbone fisso [cite: 262, 312]
    model.mtp_head.train()
    model.backbone.eval()
    
    # Ottimizzatore Adam con learning rate fisso 3e-4 [cite: 271]
    optimizer = Adam(model.mtp_head.parameters(), lr=3e-4)

    for epoch in range(1): 
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device) # Labels filtrate per l'assistente 
            
            optimizer.zero_grad()
            
            attention_mask = batch['attention_mask'].to(device)
            
            # Utilizziamo il metodo forward di MTP_LLM (mtp_llm.py)
            # Il distacco (detach) del backbone avviene internamente in MTP_LLM.forward
            _, mtp_logits = model(input_ids, attention_mask=attention_mask)
            
            # Calcolo della loss modulare
            loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # Aggiornamento della progress bar
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completata. Loss media: {avg_loss:.4f}")

    return model