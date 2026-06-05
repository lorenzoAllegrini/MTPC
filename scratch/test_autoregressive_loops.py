import sys
sys.modules['torchvision'] = None
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from utils import load_tulu_dataset, CHAT_TEMPLATE

def main():
    MODEL_ID = "google/byt5-small"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load verifier model (base + LoRA)
    lora_path = "/Users/lorenzoallegrini/Documents/MTP/saved_models/byt5_standard_lora_verifier/byt5_standard_lora"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()
    print("Verifier model loaded successfully!")
    
    # Load dataset with max_samples = 1000 and 0.05 split to match R
    from datasets import load_dataset
    full_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    filtered = full_dataset.filter(lambda x: x['source'] == "ai2-adapt-dev/flan_v2_converted", num_proc=4)
    filtered = filtered.shuffle(seed=42)
    filtered = filtered.select(range(min(1000, len(filtered))))
    splits = filtered.train_test_split(test_size=0.05)
    val_data = splits['test']
    
    # Run generation on the first 5 samples
    for i in range(5):
        messages = val_data[i]["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=60,
                do_sample=False
            )
            
        generated_text = tokenizer.decode(output_ids[0])
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"Target: {messages[-1]['content']}")
        print(f"Generated: {repr(generated_text)}")

if __name__ == "__main__":
    main()
