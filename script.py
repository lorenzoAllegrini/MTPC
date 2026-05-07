import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hybrid_workflow import FeatureShardConsumer
from probabilistic_circuits import FF
from training_loop import compute_mtpc_loss

def main():
    # 1. Configurazione Device (Mac M1/M2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # 2. Caricamento Features Estratte (Lazy Loading dei file .pt)
    try:
        dataset = FeatureShardConsumer(cache_dir="./feature_cache")
        print(f"Trovati {len(dataset)} shard di feature.")
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        print("Assicurati di avere la cartella 'feature_cache' scaricata da Colab in questa directory.")
        return

    # Usiamo un batch_size alto dato che gestiamo solo tensori leggeri, non il backbone
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 3. Inizializzazione della Testa MTP (FF)
    embed_dim = 1472  # d_model per google/byt5-small
    vocab_size = 384  # vocab_size per google/byt5-small
    window_size = 8
    gamma = 0.9
    
    print("Inizializzazione della testa FF...")
    mtp_head = FF(
        embedding_size=embed_dim, 
        vocabulary_size=vocab_size, 
        window_size=window_size
    ).to(device)

    # 4. Ottimizzatore
    optimizer = torch.optim.Adam(mtp_head.parameters(), lr=3e-4)

    # 5. Training Loop Ultra-Veloce
    epochs = 5
    print(f"Inizio addestramento per {epochs} epoche...")
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        mtp_head.train()
        for batch in pbar:
            # Riportiamo le feature a float32 se erano state salvate in float16
            features = batch["features"].to(device, dtype=torch.float32)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Passaggio in avanti SOLO sulla testa
            mtp_logits = mtp_head(features)
            
            # Calcolo della loss modulare
            loss = compute_mtpc_loss(mtp_logits, labels, window_size, gamma)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completata. Loss media: {avg_loss:.4f}")

    # 6. Salvataggio finale
    torch.save(mtp_head.state_dict(), "mtp_ff_head_trained.pth")
    print("Modello salvato con successo come 'mtp_ff_head_trained.pth'.")

if __name__ == "__main__":
    main()