import torch
from models.gpt2_moe import MoETransformer
from transformers import GPT2Tokenizer, GPT2Model

# 加载 GPT-2 Tiny 预训练参数
print("✅ 加载 GPT-2 Tiny 预训练参数...")
gpt2_tiny = GPT2Model.from_pretrained("gpt2")
moe_model = MoETransformer(gpt2_tiny)

# 加载 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    logits = moe_model(input_ids)
    output_ids = torch.argmax(logits, dim=-1)
    return tokenizer.decode(output_ids[0])

print(generate_text("The future of AI is"))
