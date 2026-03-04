import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Hyperparameters
# ----------------------------
vocab_size = 20
embedding_dim = 16
num_heads = 4
num_layers = 2
ff_hidden_dim = 64
max_seq_len = 10

assert embedding_dim % num_heads == 0


# ----------------------------
# Positional Encoding
# ----------------------------
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

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ----------------------------
# Multi-Head Attention (Causal)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v)

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, embed_dim)

        return self.out(attention)


# ----------------------------
# Transformer Decoder Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


# ----------------------------
# Mini GPT Model
# ----------------------------
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        self.layers = nn.ModuleList(
            [TransformerBlock() for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)
        return logits


# ----------------------------
# Test It
# ----------------------------
model = MiniGPT()

# Batch size = 1, sequence length = 5
sample_input = torch.randint(0, vocab_size, (1, 5))

logits = model(sample_input)

print("Input tokens:", sample_input)
print("Logits shape:", logits.shape)

# Get probabilities for next token
probs = F.softmax(logits[:, -1], dim=-1)

print("\nNext token probabilities:")
print(probs)

next_token = torch.argmax(probs, dim=-1)
print("\nPredicted next token:", next_token.item())
