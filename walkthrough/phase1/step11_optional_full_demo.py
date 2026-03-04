import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("=== PHASE1 MINI GPT & AI WORKFLOW DEMO ===\n")

# ----------------------------
# Step 1: Input Embeddings
# ----------------------------
print("Step 1: Input Embeddings")
vocab_size = 10
embedding_dim = 8
input_tokens = torch.tensor([0, 1, 2])
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embeddings = embedding_layer(input_tokens)
print("Input Tokens:", input_tokens)
print("Embeddings shape:", embeddings.shape)
print("Embeddings:\n", embeddings, "\n")

# ----------------------------
# Step 2: Simple Attention
# ----------------------------
print("Step 2: Attention Demo")
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
# Step 3: Vector DB Creation
# ----------------------------
print("Step 3: Vector DB Setup with SentenceTransformer")
sentences = ["Apple and orange are fruits",
             "The king is strong",
             "Cats and dogs are pets"]
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
vector_db = faiss.IndexFlatL2(sentence_embeddings.shape[1])
vector_db.add(sentence_embeddings.cpu().numpy())
print("Vector DB size:", vector_db.ntotal)
print("Sample embedding (first 3 dims of first sentence):", sentence_embeddings[0][:3], "\n")

# ----------------------------
# Step 4: Query Vector DB
# ----------------------------
print("Step 4: Query Vector DB")
query = model.encode(["Fruit juice"], convert_to_tensor=True)
D, I = vector_db.search(query.cpu().numpy(), k=2)
print("Top retrieved sentences for 'Fruit juice':")
for idx, dist in zip(I[0], D[0]):
    print(f"  - {sentences[idx]} (distance: {dist:.4f})")
print()

# ----------------------------
# Step 5: Multi-query retrieval
# ----------------------------
print("Step 5: Multi-query retrieval")
queries = ["Tell me about books", "Who are pets?", "What is strong?", "Tell me about colors"]
for q in queries:
    q_emb = model.encode([q], convert_to_tensor=True)
    D, I = vector_db.search(q_emb.cpu().numpy(), k=2)
    retrieved = [sentences[idx] for idx in I[0]]
    print(f"Query: {q}")
    print(f"  Retrieved info: {retrieved}")
print()

# ----------------------------
# Step 6-10: Mini Transformer + Decoder
# ----------------------------
print("Step 6-10: Mini GPT Transformer Demo\n")

num_heads = 2
head_dim = embedding_dim // num_heads
ff_hidden_dim = 32

# Positional Encoding
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
        return x + self.pe[:, :x.size(0)].squeeze(0)

# MultiHead Attention + TransformerBlock
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        if x.dim() == 2:  # add batch dim
            x = x.unsqueeze(0)  # shape (1, seq_len, embedding_dim)

        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv(x)  # (batch, seq_len, 3*embedding_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each (batch, seq_len, embedding_dim)

        # reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                 2)  # (batch, heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        # combine heads
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(attn), weights


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(nn.Linear(embedding_dim, ff_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(ff_hidden_dim, embedding_dim))
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, attn_weights



# Demo input
x = torch.randn(4, embedding_dim)
print("Step 6 - Input:")
print(x)

pos_enc = PositionalEncoding(embedding_dim)
x = pos_enc(x)
print("\nStep 7 - After Positional Encoding:")
print(x)

block = TransformerBlock(embedding_dim, num_heads, ff_hidden_dim)
x, attn_weights = block(x)
print("\nStep 8 - Transformer Block Output:")
print(x)

print("\nStep 9 - Attention Weights:")
print(attn_weights)

# ----------------------------
# Step 11: Mini Autoregressive Text Generation
# ----------------------------
print("\nStep 11: Mini GPT Text Generation Demo")
token_vocab = ["<pad>", "hello", "world", "wold", "woeld", "hew", "or", "ld", "h", "e"]
token2id = {tok: i for i,tok in enumerate(token_vocab)}

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

# Toy training loop
input_ids = torch.tensor([[1,2,3,4]])
target_ids = torch.tensor([[2,3,4,5]])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, len(token_vocab)), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generation
with torch.no_grad():
    gen_ids = [1]
    for _ in range(10):
        x_in = torch.tensor([gen_ids])
        logits = model(x_in)
        next_token = torch.argmax(logits[0,-1]).item()
        gen_ids.append(next_token)
    generated_text = [token_vocab[i] for i in gen_ids]
    print("Generated Text:", " ".join(generated_text))
