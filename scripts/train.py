import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
from models.gpt2_moe import MoETransformer

# 加载 GPT-2 Tiny 预训练模型
print("✅ 加载 GPT-2 Tiny 预训练参数...")
gpt2_tiny = GPT2Model.from_pretrained("gpt2")
moe_model = MoETransformer(gpt2_tiny)

# 加载 Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 示例文本
text = "Hello, how are you?"
input_ids = tokenizer.encode(text, return_tensors="pt")
input_ids = input_ids[:, :10]

# 训练设置
optimizer = torch.optim.Adam(moe_model.parameters(), lr=1e-3)

print("✅ 开始训练 MoE Transformer...")
for epoch in range(5):
    logits = moe_model(input_ids)
    target_ids = input_ids[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50257), target_ids.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("🎉 训练完成！")
