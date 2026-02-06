import faiss
import numpy as np

DIM = 384  # MiniLM output size

index = faiss.IndexFlatIP(DIM)  # cosine similarity
user_vectors = {}  # rollno -> vector
faiss_id_to_user = {}

def add_user_vector(rollno: str, vector: np.ndarray):
    faiss_id = index.ntotal
    index.add(np.array([vector]))
    user_vectors[rollno] = vector
    faiss_id_to_user[faiss_id] = rollno

def search(vector: np.ndarray, k: int):
    D, I = index.search(np.array([vector]), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((faiss_id_to_user[idx], float(score)))
    return results
