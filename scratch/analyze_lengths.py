import sys
sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import clean_content, CHAT_TEMPLATE

def main():
    MODEL_ID = "google/byt5-small"
    print(f"Loading tokenizer to get chat template parser...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    
    print("Loading full dataset...")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    
    task_filter = "ai2-adapt-dev/flan_v2_converted"
    print(f"Filtering dataset for task: '{task_filter}'...")
    filtered = dataset.filter(lambda x: x['source'] == task_filter, num_proc=4)
    total_samples = len(filtered)
    print(f"Total SFT samples: {total_samples}")
    
    # We will sample 15,000 examples for a very robust estimation (since it is fast now!)
    sample_size = min(15000, total_samples)
    print(f"Analyzing a sample of {sample_size} examples...")
    
    sampled_data = filtered.shuffle(seed=42).select(range(sample_size))
    
    user_lens = []
    assistant_lens = []
    total_lens = []
    
    # Track truncation statistics for different thresholds
    thresholds = [512, 800, 1024, 1536, 2048]
    truncation_counts = {t: 0 for t in thresholds}
    complete_truncation_counts = {t: 0 for t in thresholds}
    
    # Pre-encode templates to be 100% correct in byte lengths
    for idx in range(sample_size):
        item = sampled_data[idx]
        conversation = item['messages']
        
        # Clean contents using our exact clean_content regex
        cleaned_conv = []
        for msg in conversation:
            cleaned_conv.append({
                'role': msg['role'],
                'content': clean_content(msg['content'])
            })
            
        # Get full formatted text
        full_text = tokenizer.apply_chat_template(
            cleaned_conv, chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=False
        )
        
        # Get user prompt text only
        user_only_conv = []
        for msg in cleaned_conv:
            if msg['role'] == 'user':
                user_only_conv.append(msg)
            else:
                break
                
        user_text = tokenizer.apply_chat_template(
            user_only_conv, chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=True
        )
        
        # Compute lengths directly in bytes (1 byte = 1 token for ByT5)
        full_len = len(full_text.encode('utf-8'))
        user_len = len(user_text.encode('utf-8'))
        assistant_len = max(0, full_len - user_len)
        
        user_lens.append(user_len)
        assistant_lens.append(assistant_len)
        total_lens.append(full_len)
        
        for t in thresholds:
            if full_len > t:
                truncation_counts[t] += 1
            if user_len >= t:
                complete_truncation_counts[t] += 1
                
    # Calculate statistics
    user_lens = np.array(user_lens)
    assistant_lens = np.array(assistant_lens)
    total_lens = np.array(total_lens)
    
    print("\n" + "="*50)
    print("LENGTH ANALYSIS REPORT")
    print("="*50)
    print(f"Average total sequence length: {total_lens.mean():.1f} bytes/tokens")
    print(f"Average user prompt length: {user_lens.mean():.1f} bytes/tokens")
    print(f"Average assistant response length: {assistant_lens.mean():.1f} bytes/tokens")
    
    print("\nTotal sequence length percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(total_lens, p):.1f} bytes")
        
    print("\nUser prompt length percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(user_lens, p):.1f} bytes")
        
    print("\nAssistant response length percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(assistant_lens, p):.1f} bytes")
        
    print("\nTruncation Rate by max_len threshold:")
    for t in thresholds:
        tr_rate = (truncation_counts[t] / sample_size) * 100
        comp_tr_rate = (complete_truncation_counts[t] / sample_size) * 100
        print(f"  max_len = {t:4d}: Truncated: {tr_rate:6.2f}% | Assistant completely lost: {comp_tr_rate:6.2f}%")
        
    with open("/Users/lorenzoallegrini/Documents/MTP/length_report.txt", "w") as f:
        f.write(f"total_samples={total_samples}\n")
        f.write(f"sample_size={sample_size}\n")
        f.write(f"avg_total={total_lens.mean():.2f}\n")
        f.write(f"avg_user={user_lens.mean():.2f}\n")
        f.write(f"avg_assistant={assistant_lens.mean():.2f}\n")
        for p in [50, 75, 90, 95, 99]:
            f.write(f"p{p}_total={np.percentile(total_lens, p):.2f}\n")
            f.write(f"p{p}_user={np.percentile(user_lens, p):.2f}\n")
            f.write(f"p{p}_assistant={np.percentile(assistant_lens, p):.2f}\n")
        for t in thresholds:
            tr_rate = (truncation_counts[t] / sample_size) * 100
            comp_tr_rate = (complete_truncation_counts[t] / sample_size) * 100
            f.write(f"trunc_{t}={tr_rate:.4f}\n")
            f.write(f"comp_trunc_{t}={comp_tr_rate:.4f}\n")

if __name__ == "__main__":
    main()
