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

def get_preprocess_function(tokenizer, max_length, template_string):
    def preprocess_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for conversation in examples['messages']:
            full_text = tokenizer.apply_chat_template(
                conversation, chat_template=template_string, tokenize=False, add_generation_prompt=False
            )
            
            encoding = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding='max_length', add_special_tokens=False
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            labels = list(input_ids) # Copy
            
            current_offset = 0
            for msg in conversation:
                msg_text = f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
                msg_len = len(msg_text.encode('utf-8'))
                
                if msg['role'] != "assistant":
                    start = current_offset
                    end = min(current_offset + msg_len, max_length)
                    if start < max_length:
                        for i in range(start, end):
                            labels[i] = -100
                
                current_offset += msg_len
                if current_offset >= max_length: break

            # Mask padding
            for i in range(max_length):
                if attention_mask[i] == 0:
                    labels[i] = -100
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }
    return preprocess_function

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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
    
    max_len = 512
    print("Inizio pre-tokenizzazione offline (Multi-Core)...")
    
    template_str = tokenizer.chat_template
    preprocess_fn = get_preprocess_function(tokenizer, max_len, template_str)
    
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