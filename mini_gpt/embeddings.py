import torch
import torch.nn as nn


def embedding_layer_demo(vocab_size=10, embedding_dim=8):
    input_tokens = torch.tensor([0, 1, 2])
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    embeddings = embedding_layer(input_tokens)
    return input_tokens, embeddings

