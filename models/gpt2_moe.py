import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        return topk_weights, topk_indices

class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts, top_k)

    def forward(self, x):
        batch_size, hidden_dim = x.shape
        topk_weights, topk_indices = self.gating_network(x)
        output = torch.zeros_like(x)

        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = topk_indices[i, j].item()
                weight = topk_weights[i, j]
                output[i] += weight * self.experts[expert_idx](x[i])

        return output

class MoETransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, top_k=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.moe_layer = MoELayer(hidden_dim, num_experts, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.moe_layer(x))
        return x

class MoETransformer(nn.Module):
    """ GPT-2 Tiny + MoE（使用 GPT-2 预训练参数） """
    def __init__(self, gpt2_model, num_experts=4, top_k=2):
        super().__init__()

        # ✅ 继承 GPT-2 Tiny 的词嵌入层，并冻结参数
        self.embedding = gpt2_model.wte
        for param in self.embedding.parameters():
            param.requires_grad = False  # 🚀 冻结词嵌入层

        # ✅ 继承 GPT-2 Tiny 的 Transformer 层，并冻结参数
        self.num_layers = len(gpt2_model.h)
        self.layers = nn.ModuleList([
            MoETransformerBlock(gpt2_model.config.hidden_size, num_experts, top_k)
            for _ in range(self.num_layers)
        ])
        for i, layer in enumerate(self.layers):
            layer.attention.load_state_dict(gpt2_model.h[i].attn.state_dict())
            layer.norm1.load_state_dict(gpt2_model.h[i].ln_1.state_dict())
            layer.norm2.load_state_dict(gpt2_model.h[i].ln_2.state_dict())

            # 🚀 冻结 GPT-2 Tiny 预训练的参数
            for param in layer.attention.parameters():
                param.requires_grad = False
            for param in layer.norm1.parameters():
                param.requires_grad = False
            for param in layer.norm2.parameters():
                param.requires_grad = False

        # ✅ 继承 GPT-2 Tiny 的输出层，并冻结参数
        self.lm_head = gpt2_model.lm_head
        for param in self.lm_head.parameters():
            param.requires_grad = False  # 🚀 冻结输出层

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # 词嵌入
        for layer in self.layers:
            x = layer(x)  # 经过 MoE Transformer 层
        logits = self.lm_head(x)  # 语言建模输出
        return logits

