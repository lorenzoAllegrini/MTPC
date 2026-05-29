import sys
sys.modules['torchvision'] = None

import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from models.probabilistic_circuits import FF, CanonicPolyidiac, MTPC_HMM
from models.mtp_llm import MTP_LLM 
from utils import MTPChatDataset, compute_mtpc_loss, get_byt5_preprocess_function, load_tulu_dataset, CHAT_TEMPLATE

def init_emissions_from_stp(model, window_size, ranks):
    """
    Inizializza i pesi delle emissioni di qualsiasi Probabilistic Circuit
    usando la matrice STP (lm_head) del backbone.
    """
    with torch.no_grad():
        if hasattr(model.backbone, 'lm_head'):
            stp_weight = model.backbone.lm_head.weight.detach().clone()
        else:
            # Fallback se lm_head non si chiama così
            stp_weight = model.backbone.get_output_embeddings().weight.detach().clone()
            
        V, H = stp_weight.shape
        head_name = model._head_class_name

        if head_name == 'MTPC_HMM':
            # emissions weight shape: [W*R*V, H]
            emission_init = (stp_weight
                .unsqueeze(0).unsqueeze(0)
                .expand(window_size, ranks, -1, -1)
                .reshape(-1, H))
            model.heads['emissions'].weight.copy_(emission_init)
            nn.init.zeros_(model.heads['emissions'].bias)

        elif head_name == 'FF':
            for i in range(1, window_size + 1):
                layer = model.heads[f'emission_{i}']
                layer.weight.copy_(stp_weight)
                nn.init.zeros_(layer.bias)

        elif head_name == 'CanonicPolyidiac':
            emission_init = (stp_weight
                .unsqueeze(0).unsqueeze(0)
                .expand(ranks, window_size, -1, -1)
                .reshape(-1, H))
            model.heads['emission_projs'].weight.copy_(emission_init)
            nn.init.zeros_(model.heads['emission_projs'].bias)

        print(f"[INIT] Emissioni di {head_name} inizializzate dalla matrice STP ({list(stp_weight.shape)})")


def main():
    # 1. SETUP
    model_id = "google/byt5-small" # Default impostato su ByT5
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model ID: {model_id}")
    
    # Architettura assunta: Seq2Seq (ByT5)
    
    # 2. DATASET
    print("\n--- Caricamento Dataset ---")
    train_data, val_data = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=10000)
    
    # 3. PRE-TOKENIZZAZIONE OFFLINE
    max_len = 512
    print("Inizio pre-tokenizzazione offline (Multi-Core)...")
    
    print("Architettura: Encoder-Decoder (Seq2Seq). Uso AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    preprocess_fn = get_byt5_preprocess_function(tokenizer, max_len, tokenizer.chat_template)
    
    train_data = train_data.map(
        preprocess_fn,
        batched=True,
        num_proc=4,
        remove_columns=train_data.column_names,
        desc="Pre-tokenizing train data"
    )
    
    train_chat_dataset = MTPChatDataset(train_data)
    # Ottimizzazioni per l'A100: batch size alto e multi-processing per il Dataloader
    train_loader = DataLoader(
        train_chat_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    window_size = 8
    ranks = 32
    model = MTP_LLM(
        model_id=model_id, 
        head_class=MTPC_HMM, 
        window_size=window_size,
        ranks=ranks
    )

    # --- Inizializzazione emissioni dalla matrice STP di ByT5 ---
    init_emissions_from_stp(model, window_size, ranks)

    # 5. TRAINING
    model.to(device)
    model.heads.train()
    model.backbone.train() # LoRA adapters need to be trained
    
    trainable_params = list(model.heads.parameters()) + [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=0.01)
    is_log_probs = model._head_class_name in ['CanonicPolyidiac', 'MTPC_HMM']

    for epoch in range(1):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            _, mtp_logits, _ = model(input_ids, attention_mask=attention_mask)
            
            loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.9, is_log_probs=is_log_probs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/(batch_idx+1):.4f}")
        
        print(f"Epoch {epoch+1} completata. Loss media: {total_loss / len(train_loader):.4f}")
    
    # 6. SALVATAGGIO (matching R's save_model convention)
    head_type = model._head_class_name
    filename = f"mtp_head_{head_type.lower()}_w{window_size}_final.pth"
    
    # Punta alla cartella saved_models nella root del progetto (uno step sopra 'src')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    torch.save(model.heads.state_dict(), save_path)
    print(f"Training terminato e modello ({head_type}) con window_size={window_size} salvato in {save_path}!")
    
    # Save LoRA adapters
    lora_save_path = os.path.join(save_dir, f"mtp_backbone_lora_{head_type.lower()}_w{window_size}")
    model.backbone.save_pretrained(lora_save_path)
    print(f"Adattatori LoRA del backbone salvati in {lora_save_path}!")

if __name__ == "__main__":
    main()
