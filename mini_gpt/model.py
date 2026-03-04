import torch
import torch.nn as nn
from .attention import TransformerBlock

token_vocab = ["<pad>", "hello", "world", "wold", "woeld", "hew", "or", "ld", "h", "e"]


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.transformer(x)
        return self.fc(x)
