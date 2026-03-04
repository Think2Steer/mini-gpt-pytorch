import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Settings
embedding_dim = 8
num_heads = 2
head_dim = embedding_dim // num_heads

assert embedding_dim % num_heads == 0

# Dummy embeddings (like your previous step)
x = torch.randn(3, embedding_dim)  # 3 tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Combined projection for QKV (efficient way)
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)

        # Final output projection
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        seq_len = x.size(0)

        # 1️⃣ Get QKV
        qkv = self.qkv(x)  # (seq_len, 3 * embed_dim)

        # Split into Q, K, V
        qkv = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 2️⃣ Transpose for attention
        q = q.transpose(0, 1)  # (heads, seq_len, head_dim)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # 3️⃣ Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v)

        # 4️⃣ Concatenate heads
        attention = attention.transpose(0, 1).contiguous()
        attention = attention.view(seq_len, embedding_dim)

        # 5️⃣ Final projection
        output = self.out(attention)

        return output


mha = MultiHeadAttention(embedding_dim, num_heads)

output = mha(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
