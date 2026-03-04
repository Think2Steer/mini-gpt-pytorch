import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# Environment fix for Mac CPU
# ----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------
# Step 1: Corpus & Embeddings
# ----------------------------
corpus = [
    "The king is strong",
    "The queen is wise",
    "A man drives a car",
    "A woman rides a bike",
    "Cats and dogs are pets",
    "Apple and orange are fruits",
    "Bananas are yellow",
    "Strawberries are red"
]

# Sentence embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(corpus).astype("float32")

# ----------------------------
# Step 2: Vector DB (FAISS)
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"Vector DB ready: {index.ntotal} vectors, dimension: {dimension}")


# ----------------------------
# Step 3: Query Vector DB function
# ----------------------------
def retrieve_top_k(query, k=3):
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)
    return [corpus[i] for i in indices[0]]


# ----------------------------
# Step 4: Template-based agent
# ----------------------------
def agent_answer_template(query, k=3):
    top_sentences = retrieve_top_k(query, k)

    # Basic template logic: find keywords in query
    if "fruit" in query.lower():
        relevant = [s for s in top_sentences if
                    any(f in s.lower() for f in ["fruit", "apple", "banana", "orange", "strawberry"])]
        return "Fruits info: " + ", ".join(relevant)
    elif "pet" in query.lower():
        relevant = [s for s in top_sentences if any(p in s.lower() for p in ["cats", "dogs", "pets"])]
        return "Pets info: " + ", ".join(relevant)
    elif "strong" in query.lower():
        relevant = [s for s in top_sentences if "strong" in s.lower()]
        return "Strength info: " + ", ".join(relevant)
    else:
        return "Retrieved info: " + ", ".join(top_sentences)


# ----------------------------
# Step 5: Multiple queries
# ----------------------------
queries = [
    "Tell me about books",
    "Who are pets?",
    "What is strong?",
    "Tell me about colors"
]

for q in queries:
    print("\nQuery:", q)
    print("Agent Response:", agent_answer_template(q, k=3))
