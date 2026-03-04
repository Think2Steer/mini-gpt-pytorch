import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# STEP 1: Input Embeddings
# ----------------------------

seq_len = 3
embedding_dim = 8

X = torch.randn(seq_len, embedding_dim)

print("\nInput X shape:", X.shape)
print(X)

# ----------------------------
# STEP 2: Define Weight Matrices
# ----------------------------

W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

# ----------------------------
# STEP 3: Compute Q, K, V
# ----------------------------

Q = W_q(X)
K = W_k(X)
V = W_v(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# ----------------------------
# STEP 4: Compute Attention Scores
# ----------------------------
scores = torch.matmul(Q, K.T) / math.sqrt(embedding_dim)

# Create causal mask
mask = torch.tril(torch.ones(seq_len, seq_len))

# Convert zeros to -inf
mask = mask.masked_fill(mask == 0, float('-inf'))
mask = mask.masked_fill(mask == 1, 0.0)

scores = scores + mask


print("\nRaw Attention Scores:")
print(scores)

# ----------------------------
# STEP 5: Softmax
# ----------------------------

attention_weights = F.softmax(scores, dim=-1)

print("\nAttention Weights:")
print(attention_weights)

# ----------------------------
# STEP 6: Weighted Sum
# ----------------------------

output = torch.matmul(attention_weights, V)

print("\nFinal Attention Output:")
print(output)
print("Output shape:", output.shape)
