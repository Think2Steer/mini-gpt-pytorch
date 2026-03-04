import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ARTIFACT_DIR = Path(__file__).resolve().parent / "outputs" / "step03" / "artifacts"
_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
# Keep matplotlib/font cache in a writable local folder.
os.environ.setdefault("MPLCONFIGDIR", str(_ARTIFACT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_ARTIFACT_DIR / ".cache"))

# Use a non-interactive backend by default so runner logs don't hang/crash.
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Vocabulary
vocab = ["king", "queen", "man", "woman", "apple", "orange", "cat", "dog", "car", "bike"]
vocab_size = len(vocab)
embedding_dim = 5

# Step 2: Random embedding layer (like Step 1)
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Step 3: Get embeddings as numpy array
embeddings = embedding.weight.detach().numpy()

# Step 4: Reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Step 5: Plot
plt.figure(figsize=(8,6))
for i, word in enumerate(vocab):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word, fontsize=12)
plt.title("2D PCA of Random Embeddings")

parser = argparse.ArgumentParser(description="Step 03 embedding visualization")
parser.add_argument("--show", action="store_true", help="Open interactive plot window")
args = parser.parse_args()

plot_path = _ARTIFACT_DIR / "pca_embeddings.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot: {plot_path}")

if args.show:
    plt.show()
else:
    plt.close()

# Step 6: Cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)
print("Cosine similarity matrix:")
print(np.round(similarity_matrix, 2))
