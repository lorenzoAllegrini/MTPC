import torch 
import numpy as np 
from torch.utils.data import DataLoader

import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim import Adam


class MTPChatDataset(Dataset):
    def __init__(self, mapped_data):
        """
        Inizializza il dataset con i dati pre-tokenizzati (tramite dataset.map).
        """
        self.mapped_data = mapped_data

    def __len__(self):
        return len(self.mapped_data)

    def __getitem__(self, idx):
        item = self.mapped_data[idx]
        
        # Restituiamo i tensori già pronti per il DataLoader.
        # Nessuna tokenizzazione o manipolazione di stringhe avviene qui.
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }


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