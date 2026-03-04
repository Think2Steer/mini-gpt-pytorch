from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Step 1: Example corpus
corpus = [
    "The king is strong",
    "The queen is wise",
    "A man drives a car",
    "A woman rides a bike",
    "Cats and dogs are pets",
    "Apple and orange are fruits"
]

# Step 2: Load pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, works on Mac

# Step 3: Encode corpus into embeddings
embeddings = model.encode(corpus)
embeddings = np.array(embeddings).astype("float32")

# Step 4: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)

# Step 5: Query the vector DB
query = "Fruit juice"
query_embedding = model.encode([query]).astype("float32")
k = 3  # top 3 similar
distances, indices = index.search(query_embedding, k)

# Number of vectors---------
print("Number of vectors:", index.ntotal)

# Dimension
print("Dimension of vectors:", index.d)

# Print shape of embedding matrix
print("Embedding matrix shape:", embeddings.shape)
print("\nEmbedding matrix preview (first 20 dims per sentence):\n")

# Print embeddings as a "matrix table" (first 20 dims)
for i, (sentence, vec) in enumerate(zip(corpus, embeddings)):
    print(f"{i+1:2d}. {sentence}")
    print("   ", np.array2string(vec[:20], precision=4, floatmode='fixed'))
    print("---")


# Step 6: Print results
print(f"Query: {query}")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {corpus[idx]} (distance: {distances[0][i]:.4f})")
