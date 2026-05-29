import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset

sys.modules['torchvision'] = None

CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"

class MTPChatDataset(Dataset):

    def __init__(self, mapped_data):
        self.mapped_data = mapped_data

    def __len__(self):
        return len(self.mapped_data)

    def __getitem__(self, idx):
        item = self.mapped_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

def compute_mtpc_loss(mtp_logits, labels, window_size, gamma=0.9, ignore_index=-100, is_log_probs=False):
    combined_loss = 0.0
    for j in range(1, window_size + 1):
        current_logits = mtp_logits[:, :-j, j-1, :] 
        current_labels = labels[:, j:]
        
        if is_log_probs:
            step_loss = F.nll_loss(
                current_logits.reshape(-1, current_logits.size(-1)),
                current_labels.reshape(-1),
                ignore_index=ignore_index
            )
        else:
            step_loss = F.cross_entropy(
                current_logits.reshape(-1, current_logits.size(-1)),
                current_labels.reshape(-1),
                ignore_index=ignore_index
            )
        
        combined_loss += (gamma ** (j - 1)) * step_loss
        
    return combined_loss

def evabyte_encode(text, max_length):
    """
    Encodes text using EvaByte's direct UTF-8 byte encoding with offset.
    Each byte is offset by 64. BOS token (1) is prepended.
    PAD token is 0.
    """
    bytes_list = list(text.encode("utf-8"))
    input_ids = [1] + [b + 64 for b in bytes_list]  # 1 is <bos>
    
    # Truncation
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        
    # Padding
    attention_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)
        
    return input_ids, attention_mask



def get_byt5_preprocess_function(tokenizer, max_length, template_string):
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
            labels = list(input_ids)
            
            current_offset = 0
            for msg in conversation:
                prefix = f"<|{msg['role']}|>\n"
                suffix = "<|end|>\n"
                content = msg['content']
                msg_text = prefix + content + suffix
                
                msg_len = len(msg_text.encode('utf-8'))
                prefix_len = len(prefix.encode('utf-8'))
                suffix_len = len(suffix.encode('utf-8'))
                
                if msg['role'] != "assistant":
                    start = current_offset
                    end = min(current_offset + msg_len, max_length)
                    if start < max_length:
                        for i in range(start, end):
                            labels[i] = -100
                else:
                    start_prefix = current_offset
                    end_prefix = min(current_offset + prefix_len, max_length)
                    if start_prefix < max_length:
                        for i in range(start_prefix, end_prefix):
                            labels[i] = -100
                            
                    start_suffix = current_offset + msg_len - suffix_len
                    end_suffix = min(current_offset + msg_len, max_length)
                    if start_suffix < max_length:
                        for i in range(start_suffix, end_suffix):
                            labels[i] = -100
                
                current_offset += msg_len
                if current_offset >= max_length: break

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

def load_tulu_dataset(task_filter, max_samples=10000):
    full_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    print(f"Filtraggio del dataset per il task: '{task_filter}'...")
    filtered = full_dataset.filter(
        lambda x: x['source'] == task_filter,
        num_proc=4,
        desc="Filtering task"
    )
    
    filtered = filtered.shuffle(seed=42)
    actual_max = min(max_samples, len(filtered))
    filtered = filtered.select(range(actual_max))
    
    splits = filtered.train_test_split(test_size=0.01)
    return splits['train'], splits['test']

def get_grouped_params(model):
    """
    Groups parameters into 'circuit' (the MTP head) and 'lora' (the backbone adapter).
    """
    circuit_params = []
    lora_params = []
    
    # Se il modello ha un attributo _circuit (come in MTP_LLM)
    if hasattr(model, "_circuit"):
        circuit_params.extend(model._circuit.parameters())
    
    # Cerca i parametri LoRA nel backbone
    for name, param in model.backbone.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
            
    return {
        "circuit": circuit_params,
        "lora": lora_params
    }
