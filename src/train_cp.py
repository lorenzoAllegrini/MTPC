import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from utils import (
    load_tulu_dataset, 
    get_byt5_preprocess_function, 
    MTPChatDataset, 
    CHAT_TEMPLATE,
    get_grouped_params
)
from models.mtp_llm import MTP_LLM
from models.probabilistic_circuits import CanonicPolyidiac

def train_cp_phase2():
    # --- CONFIGURAZIONE ---
    MODEL_ID = "google/byt5-small"
    WINDOW_SIZE = 6
    RANKS = 32
    BATCH_SIZE = 8
    EPOCHS = 1
    PHASE2_HEAD_LR = 3e-3
    PHASE2_LLM_LR = 5e-4
    MAX_LEN = 2048
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    best_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(device)
    print(best_dtype)
    
    # Percorsi per il warm-up (FF)
    FF_LORA_PATH = f"saved_models/lora_ff_w{WINDOW_SIZE}/mtp_backbone_lora_ff_w{WINDOW_SIZE}"
    FF_HEAD_PATH = f"saved_models/mtp_head_ff_w{WINDOW_SIZE}_final.pth"
    
    # --- 1. DATASET ---
    print(f"[SYSTEM] Caricamento dataset per Fase 2 (CP)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.chat_template = CHAT_TEMPLATE
    
    train_data, val_data = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=25000)
    preprocess_fn = get_byt5_preprocess_function(tokenizer, MAX_LEN, CHAT_TEMPLATE)
    
    train_ds = train_data.map(preprocess_fn, batched=True, remove_columns=train_data.column_names)
    train_loader = DataLoader(MTPChatDataset(train_ds), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 2. MODELLO ---
    print(f"[SYSTEM] Inizializzazione MTP_LLM con testa CP (W={WINDOW_SIZE}, R={RANKS})...")
    # Carichiamo il backbone partendo dal LoRA della fase FF
    model = MTP_LLM(
        model_id=MODEL_ID, 
        head_class=CanonicPolyidiac, 
        window_size=WINDOW_SIZE, 
        ranks=RANKS, 
        lora_path=FF_LORA_PATH if os.path.exists(FF_LORA_PATH) else None
    )
    model.to(device=device, dtype=best_dtype)
    
    # --- 3. INIZIALIZZAZIONE TESTA CP DA FF ---
    if os.path.exists(FF_HEAD_PATH):
        print(f"[SYSTEM] Trasferimento pesi da FF Head: {FF_HEAD_PATH}...")
        ff_state_dict = torch.load(FF_HEAD_PATH, map_location=device)
        
        with torch.no_grad():
            H = model.embed_dim
            V = model.vocab_size
            W = WINDOW_SIZE
            R = RANKS
            
            # CP Head ha 'gate' e 'emission_projs'
            # emission_projs.weight shape: [R*W*V, H]
            new_emissions = torch.zeros(W, R, V, H, device=device, dtype=best_dtype)
            
            for t in range(W):
                # La testa FF salvata ha chiavi 'emission_1.weight', 'emission_2.weight'...
                ff_key = f'emission_{t+1}.weight'
                if ff_key in ff_state_dict:
                    ff_weight = ff_state_dict[ff_key].to(device=device, dtype=best_dtype) # [V, H]
                    # Espandiamo il peso FF su tutti i Ranks della CP
                    new_emissions[t, :, :, :] = ff_weight.unsqueeze(0).expand(R, V, H)
                else:
                    print(f"  [WARNING] Chiave {ff_key} non trovata nello state_dict FF.")
            
            # Carichiamo i pesi nel layer emission_projs della CP
            # Dobbiamo permutare e fare reshape per matchare il Linear layer (Output, Input)
            # emissions.view(B, S, R, W, V) implica che l'output del linear è R*W*V
            # La sequenza di view in probabilistic_circuits.py è (R, W, V)
            reshaped_emissions = new_emissions.permute(1, 0, 2, 3).reshape(-1, H)
            
            # Symmetry Breaking: aggiungiamo un piccolo rumore per evitare che i Rank siano identici
            noise = torch.randn_like(reshaped_emissions) * 1e-3
            reshaped_emissions += noise
            
            model._circuit.emission_projs.weight.copy_(reshaped_emissions)
            nn.init.zeros_(model._circuit.emission_projs.bias)
            
            # Inizializziamo il gate in modo neutro (uniforme)
            nn.init.zeros_(model._circuit.gate.weight)
            nn.init.zeros_(model._circuit.gate.bias)
            
        print("  [OK] Pesi FF trasferiti con successo alla testa CP.")
    else:
        print(f"[WARNING] Pesi FF Head non trovati a {FF_HEAD_PATH}. Inizializzazione casuale.")

    # --- 4. OPTIMIZER ---
    circuit_params = list(model._circuit.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': circuit_params, 'lr': PHASE2_HEAD_LR},
        {'params': lora_params, 'lr': PHASE2_LLM_LR}
    ])
    
    # Flag per la loss (CP usa log-probs tramite logsumexp)
    is_log_probs = True 
    
    # --- 5. TRAINING LOOP ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"CP Phase 2 - Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (manuale come su Colab)
            _, mtp_logits, _ = model(input_ids, attention_mask=attention_mask)
            
            # Calcolo Loss
            from utils import compute_mtpc_loss
            loss = compute_mtpc_loss(
                mtp_logits, 
                labels, 
                window_size=WINDOW_SIZE, 
                gamma=0.9, 
                is_log_probs=is_log_probs
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/(batch_idx+1):.4f}")

            
    # --- 6. SALVATAGGIO ---
    SAVE_DIR = f"saved_models/lora_cp_w{WINDOW_SIZE}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Salviamo LoRA
    model.backbone.save_pretrained(os.path.join(SAVE_DIR, f"mtp_backbone_lora_cp_w{WINDOW_SIZE}"))
    
    # Salviamo la testa CP
    head_save_path = f"saved_models/mtp_head_cp_w{WINDOW_SIZE}_final.pth"
    torch.save(model._circuit.state_dict(), head_save_path)
    
    print(f"\n[SYSTEM] Training CP terminato. Modello salvato in {SAVE_DIR} e {head_save_path}")

if __name__ == "__main__":
    train_cp_phase2()
