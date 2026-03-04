import torch
from mini_gpt.model import MiniGPT

token_vocab = ["<pad>", "hello", "world", "wold", "woeld", "hew", "or", "ld", "h", "e"]

model = MiniGPT(len(token_vocab), 8, 2, 32)

input_ids = torch.tensor([[1, 2, 3, 4]])
target_ids = torch.tensor([[2, 3, 4, 5]])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, len(token_vocab)), target_ids.view(-1))
    loss.backward()
    optimizer.step()

print("Training complete.")
