import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

class GatingNetwork(nn.Module):
    """ 门控网络：根据输入选择合适的专家 """
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
    """ MoE 层 """
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
    """ GPT-2 Transformer 层，FFN 替换为 MoE """
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
    """ GPT-2 Tiny + MoE """
    def __init__(self, vocab_size, hidden_dim, num_layers=4, num_experts=4, top_k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MoETransformerBlock(hidden_dim, num_experts, top_k) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        return logits
