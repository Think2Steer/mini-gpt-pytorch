import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # shape (1, max_len, embedding_dim)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:, :seq_len].squeeze(0)


# Settings
embedding_dim = 8
num_heads = 2
head_dim = embedding_dim // num_heads
ff_hidden_dim = 32  # usually 4x embedding size

assert embedding_dim % num_heads == 0


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        seq_len = x.size(0)

        qkv = self.qkv(x)
        qkv = qkv.view(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v)

        attention = attention.transpose(0, 1).contiguous()
        attention = attention.view(seq_len, embedding_dim)

        return self.out(attention)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # 1️⃣ Attention + Residual
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # 2️⃣ Feed Forward + Residual
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x


# Dummy input (3 tokens, embedding size 8)
x = torch.randn(3, embedding_dim)

pos_encoder = PositionalEncoding(embedding_dim)
x = pos_encoder(x)


block = TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("\nFinal Output:")
print(output)
