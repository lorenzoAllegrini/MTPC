import sys
sys.modules['torchvision'] = None
# Assicurati che i percorsi siano corretti per il tuo ambiente
sys.path.append("/Users/lorenzoallegrini/Documents/MTP_PoC")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader

from probabilistic_circuits import FF
from mtp_llm import MTP_LLM 
from training_loop import MTPChatDataset, train_mtp

def load_tulu_splits():
    full_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    full_dataset = full_dataset.shuffle(seed=42)
    
    splits = full_dataset.train_test_split(test_size=0.01)
    
    return splits['train'], splits['test']

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_data, val_data = load_tulu_splits()
    print(f"Dataset caricato: {len(train_data)} train, {len(val_data)} val")
    
    model_id = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|assistant|>\\n' }}"
        "{% endif %}"
    )
    
    train_chat_dataset = MTPChatDataset(train_data, tokenizer, max_length=512)
    # Ottimizzazioni per l'A100: batch size alto e multi-processing per il Dataloader
    train_loader = DataLoader(
        train_chat_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    window_size = 8
    model = MTP_LLM(
        model_id=model_id, 
        head_class=FF, 
        window_size=window_size
    )

    train_mtp(
        model=model, 
        train_loader=train_loader, 
        device=device, 
        window_size=window_size, 
        gamma=0.9
    )
    
    # 6. SALVATAGGIO
    torch.save(model.mtp_head.state_dict(), "mtp_head_byt5_final.pth")
    print("Training terminato con successo!")