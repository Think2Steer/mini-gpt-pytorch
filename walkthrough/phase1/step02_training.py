import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Tiny corpus: simple pairs (target -> context)
corpus = [
    ("king", "queen"),
    ("man", "woman"),
    ("apple", "orange"),
    ("cat", "dog"),
    ("car", "bike")
]

# Sorted vocabulary keeps index mapping stable across runs.
vocab = sorted({w for pair in corpus for w in pair})
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

vocab_size = len(vocab)
embedding_dim = 5

# Embedding layer for input and output (skip-gram style)
input_embeddings = nn.Embedding(vocab_size, embedding_dim)
output_embeddings = nn.Embedding(vocab_size, embedding_dim)

# Optimizer
optimizer = optim.SGD(list(input_embeddings.parameters()) + list(output_embeddings.parameters()), lr=0.1)

# Training loop
for epoch in range(100):
    total_loss = 0
    for target, context in corpus:
        target_idx = torch.tensor([word2idx[target]])
        context_idx = torch.tensor([word2idx[context]])

        # Forward: predict context word from target word
        v_target = input_embeddings(target_idx)

        # Compare target embedding against every output embedding.
        # This gives one logit per vocabulary token.
        scores = torch.matmul(v_target, output_embeddings.weight.t())  # (1, vocab_size)
        log_probs = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(log_probs, context_idx)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Inspect learned embeddings
print("\nLearned input embeddings:")
for i, word in idx2word.items():
    print(f"{word} -> {input_embeddings(torch.tensor([i])).detach().numpy()}")
