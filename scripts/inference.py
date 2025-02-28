import torch
from models.gpt2_moe import MoETransformer
from transformers import GPT2Tokenizer

vocab_size = 50257
hidden_dim = 128
num_layers = 4
num_experts = 4
top_k = 2

model = MoETransformer(vocab_size, hidden_dim, num_layers, num_experts, top_k)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    logits = model(input_ids)
    output_ids = torch.argmax(logits, dim=-1)
    return tokenizer.decode(output_ids[0])

print(generate_text("The future of AI is"))
