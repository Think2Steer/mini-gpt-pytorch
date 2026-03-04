import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

print("=== PHASE1 MINI GPT & AI WORKFLOW DEMO (Full) ===\n")

# ----------------------------
# Step 1: Input Embeddings
# ----------------------------
print("Step 1: Input Embeddings")
vocab_size = 10
embedding_dim = 8
input_tokens = torch.tensor([0, 1, 2])  # toy token ids
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embeddings = embedding_layer(input_tokens)
print("Input Tokens:", input_tokens)
print("Embeddings shape:", embeddings.shape)
print("Embeddings:\n", embeddings, "\n")

# ----------------------------
# Step 2: Simple Attention Demo
# ----------------------------
print("Step 2: Simple Attention")
seq_len = embeddings.shape[0]
W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

Q = W_q(embeddings)
K = W_k(embeddings)
V = W_v(embeddings)

scores = torch.matmul(Q, K.T) / math.sqrt(embedding_dim)
weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(weights, V)
print("Attention Weights:\n", weights)
print("Attention Output:\n", attn_output, "\n")

# ----------------------------
# Step 6-10: Mini Transformer
# ----------------------------
print("Step 6-10: Mini Transformer Block Demo\n")

num_heads = 2
ff_hidden_dim = 32

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0)/embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim*3, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, embedding_dim]
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)  # [batch, seq_len, 3*embedding_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each [batch, seq_len, embedding_dim]

        # reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores = scores.masked_fill(mask==0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        attn = torch.matmul(weights, v)
        attn = attn.transpose(1,2).contiguous().view(batch_size, seq_len, -1)

        return self.out(attn), weights

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
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights

# Dummy batch input
x = torch.randn(1, 4, embedding_dim)  # [batch, seq_len, embedding_dim]
print("Step 6 - Initial Input:\n", x)

pos_encoder = PositionalEncoding(embedding_dim)
x = pos_encoder(x)
print("\nStep 7 - After Positional Encoding:\n", x)

block = TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)
x, attn_weights = block(x)
print("\nStep 8 - Transformer Output:\n", x)
print("\nStep 9 - Attention Weights:\n", attn_weights)

# ----------------------------
# Step 11: Mini GPT Text Generation
# ----------------------------
print("\nStep 11 - Mini GPT Text Generation Demo")
token_vocab = ["<pad>", "hello", "world", "wold", "woeld", "hew", "or", "ld", "h", "e"]
token2id = {tok: i for i, tok in enumerate(token_vocab)}

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

model = MiniGPT(len(token_vocab), embedding_dim, num_heads, ff_hidden_dim)

# Toy input
input_ids = torch.tensor([[1,2,3,4]])
target_ids = torch.tensor([[2,3,4,5]])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(0, 301, 50):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, len(token_vocab)), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generation
with torch.no_grad():
    gen_ids = [1]  # start token
    for _ in range(10):
        x_in = torch.tensor([gen_ids])
        logits = model(x_in)
        next_token = torch.argmax(logits[0,-1]).item()
        gen_ids.append(next_token)

    generated_text = [token_vocab[i] for i in gen_ids]
    print("Generated Text:", " ".join(generated_text))

# ----------------------------
# Step 12: Attention Heatmap
# ----------------------------
plt.figure(figsize=(6,4))
plt.title("Attention Head 0 (batch 0)")
plt.imshow(attn_weights[0,0].detach().numpy(), cmap="viridis")
plt.colorbar()
plt.show()
