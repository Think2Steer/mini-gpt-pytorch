import torch
import torch.nn as nn

# Step 1: Define a tiny vocabulary
vocab = ["king", "queen", "man", "woman", "apple", "orange", "cat", "dog", "car", "bike"]
vocab_size = len(vocab)
embedding_dim = 5  # small for demonstration

# Step 2: Create an embedding layer
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Step 3: Lookup embeddings for each word
for i, word in enumerate(vocab):
    vector = embedding(torch.tensor(i))
    print(f"{word} -> {vector.tolist()}")

# Step 4: Inspect full embedding weight matrix
print("\nFull embedding weight matrix:")
print(embedding.weight)
print("Shape:", embedding.weight.shape)
