import numpy as np
from embedder import embed_text
from weights import QUESTION_WEIGHTS

def create_weighted_vector(responses: dict) -> np.ndarray:
    final_vector = None

    for q_id, answer in responses.items():
        weight = QUESTION_WEIGHTS.get(q_id, 1.0)
        emb = embed_text(answer) * weight
        final_vector = emb if final_vector is None else final_vector + emb

    # normalize final vector
    norm = np.linalg.norm(final_vector)
    return final_vector / norm
