import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("\n==============================")
print("STEP 1: Define Corpus")
print("==============================")

corpus = [
    "Apple and orange are fruits",
    "The king is strong",
    "The queen is wise",
    "Cats and dogs are pets",
    "Red and blue are colors"
]

for i, sentence in enumerate(corpus):
    print(f"{i}: {sentence}")

# ---------------------------------------------------
print("\n==============================")
print("STEP 2: Generate Embeddings")
print("==============================")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(corpus)

print("Embedding matrix shape:", embeddings.shape)
print("\nFirst 2 embedding vectors (truncated):")

for i in range(2):
    print(f"\nSentence: {corpus[i]}")
    print(embeddings[i][:10])  # show first 10 dimensions only

# ---------------------------------------------------
print("\n==============================")
print("STEP 3: Store in FAISS Vector DB")
print("==============================")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Total vectors stored in FAISS:", index.ntotal)

# ---------------------------------------------------
print("\n==============================")
print("STEP 4: Query Processing")
print("==============================")

query = "Tell me about books"
print("Query:", query)

query_embedding = model.encode([query])

print("\nQuery embedding (first 10 dims):")
print(query_embedding[0][:10])

distances, indices = index.search(np.array(query_embedding), k=3)

print("\nSimilarity distances:")
print(distances)

print("\nTop matched indices:")
print(indices)

retrieved_sentences = [corpus[i] for i in indices[0]]

print("\nRetrieved Sentences:")
for sentence in retrieved_sentences:
    print("-", sentence)

# ---------------------------------------------------
print("\n==============================")
print("STEP 5: Decision Logic / Response")
print("==============================")

distance_threshold = 1.2

if distances[0][0] > distance_threshold:
    response = "No relevant information found in corpus."
else:
    response = "Based on retrieval: " + ", ".join(retrieved_sentences)

print("Final Agent Response:")
print(response)

print("\n==============================")
print("PIPELINE COMPLETE")
print("==============================")
