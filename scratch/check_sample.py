import sys
sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from utils import load_tulu_dataset

def main():
    train_data, val_data = load_tulu_dataset("ai2-adapt-dev/flan_v2_converted", max_samples=15000)
    
    found = False
    for dataset_name, dataset in [("train", train_data), ("test", val_data)]:
        for idx, item in enumerate(dataset):
            messages = item['messages']
            user_msg = messages[0]['content']
            if "chopping coconuts" in user_msg:
                print(f"Found sample in {dataset_name} at index {idx}!")
                print(f"User message:\n{user_msg}")
                print(f"Assistant message:\n{messages[1]['content']}")
                found = True
                break
        if found:
            break
            
    if not found:
        print("Sample not found in 15000 items.")

if __name__ == "__main__":
    main()
