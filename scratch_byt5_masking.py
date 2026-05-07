from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\\n' }}"
    "{% endif %}"
)

conversation = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm a ByT5 model!"}
]

full_text = tokenizer.apply_chat_template(conversation, tokenize=False)
input_ids = tokenizer.encode(full_text, truncation=True, max_length=512, padding='max_length', add_special_tokens=False)

labels = list(input_ids)
current_offset = 0

for message in conversation:
    role = message['role']
    msg_text = f"<|{role}|>\n{message['content']}<|end|>\n"
    msg_len = len(msg_text.encode('utf-8'))
    
    if role != "assistant":
        for i in range(current_offset, min(current_offset + msg_len, 512)):
            labels[i] = -100
            
    current_offset += msg_len

# Print words mapped to labels to check masking
for i in range(122):
    print(f"Token: {input_ids[i]}, Char: {chr(input_ids[i]-3) if input_ids[i]>3 else ''}, Label: {labels[i]}")
