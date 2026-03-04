import torch
import torch.nn as nn

# ----------------------------
# STEP 1: Tiny Vocabulary
# ----------------------------

vocab = {
    "I": 0,
    "love": 1,
    "AI": 2,
    "and": 3,
    "ML": 4
}

vocab_size = len(vocab)
embedding_dim = 8   # small for visualization

print("Vocabulary size:", vocab_size)
print("Embedding dimension:", embedding_dim)

# ----------------------------
# STEP 2: Create Embedding Layer
# ----------------------------

embedding = nn.Embedding(vocab_size, embedding_dim)

# ----------------------------
# STEP 3: Example Input Sentence
# ----------------------------

sentence = ["I", "love", "AI"]
token_ids = torch.tensor([vocab[word] for word in sentence])

print("\nToken IDs:", token_ids)

# ----------------------------
# STEP 4: Get Embeddings
# ----------------------------

embedded_tokens = embedding(token_ids)

print("\nEmbedding Matrix Shape:", embedded_tokens.shape)
print("\nEmbeddings:")
print(embedded_tokens)
