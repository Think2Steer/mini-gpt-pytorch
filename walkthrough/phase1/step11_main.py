import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Tiny Training Text
# ----------------------------
text = "hello world hello gpt hello world"

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return ''.join([itos[i] for i in t])

data = encode(text)

# ----------------------------
# Hyperparameters
# ----------------------------
embedding_dim = 32
num_heads = 4
num_layers = 2
ff_hidden_dim = 128
block_size = 8
learning_rate = 1e-3
epochs = 300

device = "cpu"

# ----------------------------
# Model Components
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T)).to(device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, v)

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(B, T, C)

        return self.out(attention)


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


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock() for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T))

        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# ----------------------------
# Training Setup
# ----------------------------
model = MiniGPT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create training batches
def get_batch():
    i = torch.randint(0, len(data) - block_size, (1,))
    x = data[i:i+block_size]
    y = data[i+1:i+block_size+1]
    return x.unsqueeze(0), y.unsqueeze(0)


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# ----------------------------
# Text Generation
# ----------------------------
context = torch.zeros((1, 1), dtype=torch.long)

generated = []

for _ in range(50):
    context_cond = context[:, -block_size:]  # 👈 crop

    logits, _ = model(context_cond)
    probs = F.softmax(logits[:, -1, :], dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)

    context = torch.cat((context, next_token), dim=1)
    generated.append(next_token.item())


print("\nGenerated Text:")
print(decode(generated))
