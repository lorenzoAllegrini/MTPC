import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class MTP_CoreModel(nn.Module):
    """
    The Wrapper model containing the backbone (e.g., T5) and the mtp_head.
    Provides methods to decouple feature extraction from head training.
    """
    def __init__(self, backbone: nn.Module, mtp_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.mtp_head = mtp_head
        
    def extract_decoder_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Runs the backbone in eval mode using decoder causal forcing (decoder_input_ids=input_ids).
        Returns the detached hidden states from the last decoder layer.
        """
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=input_ids,
                output_hidden_states=True
            )
        # Extract the last layer of the decoder and detach from computation graph
        hidden_states = outputs.decoder_hidden_states[-1].detach()
        return hidden_states

    def forward_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Passes the pre-computed/detached features through the MTP head.
        """
        return self.mtp_head(hidden_states)


class FeatureShardProducer:
    """
    The Extractor. Runs the backbone over a DataLoader and saves the detached features 
    and labels into sharded .pt files. Designed to run on a Cloud GPU.
    """
    def __init__(self, model: MTP_CoreModel, device: torch.device, output_dir: str | Path, precision: torch.dtype = torch.float32):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.precision = precision
        
        self.model.to(self.device)
        self.model.eval()

    def process_dataset(self, dataloader: DataLoader):
        """
        Iterates through the dataloader. Extracts features using the core model and 
        saves each batch to a separate .pt file.
        Note: It is recommended to use batch_size=1 in the dataloader to save single sequences.
        """
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Features")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Use autocast if on supported hardware for faster/lighter extraction
            autocast_device = self.device.type if self.device.type in ['cuda', 'mps', 'cpu'] else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=self.precision):
                features = self.model.extract_decoder_features(input_ids, attention_mask)
            
            # Move to CPU and cast to precision to prevent VRAM accumulation and save disk space
            shard_data = {
                "features": features.cpu().to(self.precision),
                "labels": labels.cpu()
            }
            
            shard_path = self.output_dir / f"shard_{batch_idx:06d}.pt"
            torch.save(shard_data, shard_path)


class FeatureShardConsumer(Dataset):
    """
    The Dataset for the extracted features. Loads the sharded .pt files.
    Designed to be used on a Local Machine for lightweight head training.
    """
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        
        # Securely load and sort the file paths using pathlib
        self.shard_paths = sorted(list(self.cache_dir.glob("shard_*.pt")))
        
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.shard_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Loads the .pt file, removes any artificial batch dimensions (assuming Producer 
        used batch_size=1), and returns the tensors.
        """
        shard_path = self.shard_paths[idx]
        
        # weights_only=True is a security best practice for torch.load
        shard_data = torch.load(shard_path, weights_only=True)
        
        # Remove the batch dimension added during extraction (assuming B=1)
        # This allows the new DataLoader to dynamically re-batch the samples.
        features = shard_data["features"].squeeze(0)
        labels = shard_data["labels"].squeeze(0)
        
        return {
            "features": features,
            "labels": labels
        }


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Snippet A: Feature Extraction (e.g., Run on a Cloud GPU)
    # ---------------------------------------------------------
    print("--- Boilerplate: Feature Extraction ---")
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    import torch
    
    # Importiamo le classi dal tuo progetto
    from probabilistic_circuits import FF
    from training_loop import MTPChatDataset
    
    # 1. Carica Dataset & Tokenizer
    model_id = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Template identico a quello che usi in script.py
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|assistant|>\\n' }}"
        "{% endif %}"
    )
    
    print("Scaricamento/Caricamento del dataset tulu-3-sft-mixture...")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    # Utilizziamo la classe Dataset refattorizzata con Lazy Loading
    chat_dataset = MTPChatDataset(raw_data=dataset, tokenizer=tokenizer, max_length=512)
    
    # IMPORTANTE: batch_size=1 affinché ogni shard (.pt) rappresenti esattamente 1 sequenza indipendente
    extraction_loader = DataLoader(chat_dataset, batch_size=1, shuffle=False)
    
    # 2. Inizializza il backbone e la testa
    print("Inizializzazione dei modelli...")
    backbone = T5ForConditionalGeneration.from_pretrained(model_id)
    embed_dim = backbone.config.d_model
    vocab_size = backbone.config.vocab_size
    window_size = 8
    
    mtp_head = FF(embedding_size=embed_dim, vocabulary_size=vocab_size, window_size=window_size)
    core_model = MTP_CoreModel(backbone, mtp_head)
    
    # 3. Setup e Avvio del Producer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device per l'estrazione: {device}")
    
    producer = FeatureShardProducer(
        model=core_model,
        device=device,
        output_dir="./feature_cache",
        # Usa float16 se sei su CUDA per dimezzare lo spazio disco e accelerare I/O.
        # Usa float32 per MPS per compatibilità sicura.
        precision=torch.float16 if device.type == "cuda" else torch.float32 
    )
    
    print("Pronto per estrarre le feature!")
    # Scommenta la riga sottostante per avviare l'estrazione (ATTENZIONE: richiederà tempo e spazio su disco)
    # producer.process_dataset(extraction_loader)

    # ---------------------------------------------------------
    # Snippet B: Head Training Loop (e.g., Run on Local M1 Max)
    # ---------------------------------------------------------
    print("--- Boilerplate: Head Training Loop ---")
    """
    # 1. Load the pre-extracted features
    consumer_dataset = FeatureShardConsumer(cache_dir="./feature_cache")
    
    # The DataLoader will dynamically re-batch the single-sequence shards.
    # You can now use a large batch size without OOM because the backbone is gone!
    train_loader = DataLoader(consumer_dataset, batch_size=64, shuffle=True)
    
    # 2. Initialize ONLY the head 
    mtp_head = FF(...) 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mtp_head.to(device)
    
    optimizer = torch.optim.Adam(mtp_head.parameters(), lr=3e-4)
    
    # 3. Ultra-Fast Training Loop
    for epoch in range(1):
        for batch in train_loader:
            features = batch["features"].to(device, dtype=torch.float32) # Cast back to float32 for training if needed
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Pass directly through the head
            mtp_logits = mtp_head(features)
            
            # loss = compute_mtpc_loss(mtp_logits, labels, ...)
            # loss.backward()
            # optimizer.step()
    """
