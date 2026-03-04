import torch
from mini_gpt.embeddings import embedding_layer_demo
from mini_gpt.attention import PositionalEncoding, TransformerBlock
from mini_gpt.model import MiniGPT, token_vocab
from retrieval.vector_db import run_vector_db_demo
import torch.nn as nn

print("=== PHASE1 MINI GPT & AI WORKFLOW DEMO ===\n")

# Step 1
input_tokens, embeddings = embedding_layer_demo()
print("Step 1 Embeddings:\n", embeddings, "\n")

# Step 2-5
run_vector_db_demo()

# Step 6-10 Mini Transformer Demo
embedding_dim = embeddings.shape[1]
num_heads = 2
ff_hidden_dim = 32

x = torch.randn(4, embedding_dim)
print("\nStep 6 - Input:")
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

# Step 11: Mini GPT Text Generation
model = MiniGPT(len(token_vocab), embedding_dim, num_heads, ff_hidden_dim)

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

with torch.no_grad():
    gen_ids = [1]
    for _ in range(10):
        x_in = torch.tensor([gen_ids])
        logits = model(x_in)
        next_token = torch.argmax(logits[0,-1]).item()
        gen_ids.append(next_token)
    generated_text = [token_vocab[i] for i in gen_ids]
    print("Generated Text:", " ".join(generated_text))
