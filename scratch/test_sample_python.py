import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append("/Users/lorenzoallegrini/Documents/MTP/src")
from models.probabilistic_circuits import CanonicPolyidiac
from models.mtp_llm import MTP_LLM
from utils import CHAT_TEMPLATE

def main():
    MODEL_ID = "google/byt5-small"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    save_dir = "/Users/lorenzoallegrini/Documents/MTP/saved_models"
    lora_path = os.path.join(save_dir, "byt5_standard_lora_verifier", "byt5_standard_lora")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    
    # Load model (as verifier standard model)
    from peft import PeftModel
    base_model = torch.hub.load("huggingface/pytorch-transformers", "model", MODEL_ID) if False else None
    from transformers import AutoModelForSeq2SeqLM
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()
    
    prompt_text = (
        "<|user|>\n"
        "In this task, you will be presented with a question, a word, and a POS tag. You have to determine whether the part-of-speech tag of the given word in the question is equal to the given POS tag or not. Give your answer with True or False. Here is the Alphabetical list of part-of-speech tags used in this task: CC: Coordinating conjunction, CD: Cardinal number, DT: Determiner, EX: Existential there, FW: Foreign word, IN: Preposition or subordinating conjunction, JJ: Adjective, JJR: Adjective, comparative, JJS: Adjective, superlative, LS: List item marker, MD: Modal, NN: Noun, singular or mass, NNS: Noun, plural, NNP: Proper noun, singular, NNPS: Proper noun, plural, PDT: Predeterminer, POS: Possessive ending, PRP: Personal pronoun, PRP$: Possessive pronoun, RB: Adverb, RBR: Adverb, comparative, RBS: Adverb, superlative, RP: Particle, SYM: Symbol, TO: to, UH: Interjection, VB: Verb, base form, VBD: Verb, past tense, VBG: Verb, gerund or present participle, VBN: Verb, past participle, VBP: Verb, non-3rd person singular present, VBZ: Verb, 3rd person singular present, WDT: Wh-determiner, WP: Wh-pronoun, WP$: Possessive wh-pronoun, WRB: Wh-adverb\n"
        "Q: The second oldest monument found in the Oise region included work by a sculpture artist most famous for which other statue ? \n"
        ", Word: found \n"
        ", POS tag: JJ\n"
        "A: \n"
        "<|assistant|>\n"
    )
    pfx = "False"
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    initial_decoder_ids = tokenizer.encode(pfx, add_special_tokens=False, return_tensors="pt").to(device)
    
    P = prompt_ids.shape[1]
    
    # We test two states:
    # State 1: decoder_ids = [zeros(P+1), "False"]
    # We want to see what is predicted after "False"
    decoder_ids = torch.zeros((1, P + 1), dtype=torch.long, device=device)
    decoder_ids = torch.cat([decoder_ids, initial_decoder_ids], dim=1)
    
    encoder_input_ids = torch.cat([prompt_ids, initial_decoder_ids], dim=1)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoder_input_ids,
            decoder_input_ids=decoder_ids,
            use_cache=False
        )
        logits = outputs.logits
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top = torch.topk(probs, 5)
        
    print("\n--- Predictions after 'False' ---")
    for idx, (val, tok) in enumerate(zip(top.values, top.indices)):
        char = tokenizer.decode([tok.item()])
        print(f"  Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")
        
    # State 2: decoder_ids = [zeros(P+1), "False "] (with space)
    # We want to see what is predicted after "False "
    space_tensor = torch.tensor([[35]], dtype=torch.long, device=device)
    decoder_ids_space = torch.cat([decoder_ids, space_tensor], dim=1)
    encoder_input_ids_space = torch.cat([encoder_input_ids, space_tensor], dim=1)
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoder_input_ids_space,
            decoder_input_ids=decoder_ids_space,
            use_cache=False
        )
        logits = outputs.logits
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top = torch.topk(probs, 5)
        
    print("\n--- Predictions after 'False ' ---")
    for idx, (val, tok) in enumerate(zip(top.values, top.indices)):
        char = tokenizer.decode([tok.item()])
        print(f"  Top {idx+1}: {repr(char)} (p={val.item():.6f}, id={tok.item()})")

if __name__ == "__main__":
    main()
