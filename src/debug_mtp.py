import torch
from transformers import AutoTokenizer, AutoConfig
from models.mtp_llm import MTP_LLM
from models.probabilistic_circuits import CanonicPolyidiac
from utils import CHAT_TEMPLATE

def debug_generation(prompt, prefix, model_id="google/byt5-small", weights_path="saved_models/mtp_head_cp_w6_final.pth", lora_path="saved_models/lora_cp_w6/mtp_backbone_lora_cp_w6"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDEBUG GENERATION - Head CP Window 6")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = MTP_LLM(
        model_id=model_id,
        head_class=CanonicPolyidiac,
        window_size=6,
        lora_path=lora_path
    )
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.heads.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Encode prompt (for encoder)
    encoder_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Encode prefix (for decoder)
    decoder_start_token_id = model.backbone.config.decoder_start_token_id
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(device)
    decoder_input_ids = torch.cat([
        torch.tensor([[decoder_start_token_id]], device=device),
        prefix_ids
    ], dim=1)

    full_prompt = prompt + prefix

    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model.backbone(
            input_ids=encoder_input_ids, 
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )
        last_hidden_state = outputs.decoder_hidden_states[-1] # [B, L, D]
        
        # Generate draft (GREEDY)
        # 1. Select most probable Rank
        gate_logits = model.heads.gate(last_hidden_state[:, -1:, :]) # [1, 1, R]
        selected_rank = gate_logits.argmax(dim=-1).item()
        
        # 2. Select most probable tokens for that Rank
        flat_emissions = model.heads.emission_projs(last_hidden_state[:, -1:, :])
        emissions = flat_emissions.view(1, 1, model._circuit.ranks, model._circuit.window_size, model._circuit.vocabulary_size)
        selected_logits = emissions[0, 0, selected_rank, :, :] # [W, V]
        drafted_tokens = selected_logits.argmax(dim=-1) # [W]
        
        draft_text = tokenizer.decode(drafted_tokens)
        
        print("\n" + "="*50)
        print(f"PROMPT (end): ...{prompt[-100:]!r}")
        print(f"PREFIX: {prefix!r}")
        print(f"DRAFT PREDICTION (W=6): {draft_text!r}")
        print(f"DRAFT BYTES: {drafted_tokens.tolist()}")
        print("="*50)

if __name__ == "__main__":
    prompt_5 = """User: [QUESTION] Premise: "Two men are in a room and are looking in opposite directions."
Hypothesis: "A couple looks amorously at each other across a dinner table."
Do we know that the hypothesis entailed by the premise?
Two people can not look at each other while looking in opposite directions.
The answer is no.

Q: Can we conclude from "A boy in a jacket playing." that "A girl is playing."?
Options:
- yes
- no
- it is not possible to tell
A: In fact that there is A boy in a jacket playing does not imply that there is A girl is playing.
The answer is it is not possible to tell.

QUESTION: Premise: "A young boy is standing by a fruit stand."
Based on this premise, can we conclude that the hypothesis "A lady is choosing a melon at the fruit stand." is true?
Options:
- yes
- it is not possible to tell
- no

Let's solve it slowly: The young boy standing by the fruit stand is not a lady choosing a melon at the fruit stand.
The answer is no.

[QUESTION] Can we conclude from "This young lady with her white polka dot blouse is about to take a stroll on her bike in a foreign country." that "A young lady in a blouse rides her bike in a foreign country."?
Options:
- yes
- no
- it is not possible to tell
About to take a stroll does not mean rides her bike.
The answer is it is not possible to tell.

Q: If "A man in a black hat and blue shirt juggling." does that mean that "A man is juggling."?
Options:
- yes
- it is not possible to tell
- no
A: The man who is juggling is wearing a black hat and blue shirt.
The answer is yes.

[QUESTION] Premise: "Three people recline in massage chairs."
Hypothesis: "Three friends try out the massage chairs at the mall."
Is the hypothesis entailed by the premise?
Options:
- yes
- it is not possible to tell
- no

Assistant: </s>"""
    prefix_5 = "Not all march the world thera"
    
    debug_generation(prompt_5, prefix_5)
    
    prompt_7 = """User: [Question]It takes 50 minutes to cut a woman's hair, 15 minutes to cut a man's hair, and 25 minutes to cut a kid's hair. If Joe cut 3 women's, 2 men's, and 3 children's hair, how much time did he spend cutting hair?
[Answer]He spent 3 * 50 = 150 minutes cutting women's hair. He spent 2 * 15 = 30 minutes cutting men's hair. He spent 3 * 25 = 75 minutes cutting children's hair. He spent a total of 150 + 30 + 75 = 255 minutes. The answer is 255.
Q: Tedra is harvesting his tomatoes. He harvests 400 kg on Wednesday, and half as much on Thursday. He harvests a total of 2000 kg on Wednesday, Thursday, and Friday. How many tomatoes of the number he harvested on Friday are remaining if he gives away 700kg of them to his friends?
A: Tedra harvests 1 / 2 * 400 kg = 200 kg of tomatoes on Thursday. Tedra harvests 2000 kg - 400 kg - 200 kg = 1400 kg of tomatoes on Friday. After giving away tomatoes to his friends, he is left with 1400 kg - 700 kg = 700 kg of tomatoes. The answer is 700.
Question: Axel bought an aquarium that was marked down 50% from an original price of $120. But he also paid additional sales tax equal to 5% of the reduced price. What was the total cost of the aquarium?
Answer: The aquarium was bought for $120 x 50 / 100 = $60 less. So the marked down price of the aquarium was $120 - $60 = $60. Axel paid $60 x 5 / 100 = $3 additional for the sales tax. Therefore, the total cost of the aquarium was $60 + $3 = $63. The answer is 63.
[Question]Michelangelo has 28 ceilings to paint. This week, he paints 12 of them. Next week, he will paint 1/4 the number of ceilings he did this week. How many ceilings will be left to paint after next week?
[Answer]After this week, Michelangelo will have 28 - 12 = 16 more ceilings to paint. Next week, Michelangelo will paint 12 / 4 = 3 ceilings. Michelangelo will have 16 - 3 = 13 more ceilings to paint. The answer is 13.
Q: Billy can spit a watermelon seed 30 inches. Madison can spit a watermelon seed 20% farther than Billy. Ryan can spit a watermelon seed 50% shorter than Madison. How far can Ryan spit a watermelon seed?
A: Madison can spit a seed 20% farther than Billy's 30 inches so that's .20 * 30 = 6 inches farther. Billy spits a seed 30 inches and Madison can spit 6 inches farther for a total of 30 + 6 = 36 inches. Ryan can spit a seed 50% shorter than Madison's 36 inches for a total of .50 * 36 = 18 inches. The answer is 18.
Question: Andrew is having two of his friends over for a sleepover. For one friend, Brian, he asks his mother to buy 3 donuts. He asks for his other friend, Samuel, to get the same. A day before the sleepover Andrew ends up inviting two more friends and asks his mother to buy them the same amount of donuts. Just in case, Andrew’s mother wants to buy one more donut for each of Andrew’s friends. Andrew’s mother is going to buy the same amount of donuts for Andrew as everybody else. How many donuts will Andrew’s mother need to buy?
Answer:
Assistant: </s>"""
    prefix_7 = "Andrew is "
    
    debug_generation(prompt_7, prefix_7)
