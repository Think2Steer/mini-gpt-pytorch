from sentence_transformers import SentenceTransformer
import faiss
import torch


def run_vector_db_demo():
    sentences = ["Apple and orange are fruits",
                 "The king is strong",
                 "Cats and dogs are pets"]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    vector_db = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    vector_db.add(sentence_embeddings.cpu().numpy())

    queries = ["Fruit juice", "Tell me about books", "Who are pets?", "What is strong?"]
    for q in queries:
        q_emb = model.encode([q], convert_to_tensor=True)
        D, I = vector_db.search(q_emb.cpu().numpy(), k=2)
        retrieved = [sentences[idx] for idx in I[0]]
        print(f"Query: {q}")
        print(f"  Retrieved info: {retrieved}")
