from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from faiss_store import add_user_vector, search, user_vectors
from mappings import create_weighted_vector

app = FastAPI(title="Dating App Matching API")

# ------------------ Schemas ------------------

class RegisterRequest(BaseModel):
    user_id: str
    responses: Dict[str, str]

class MatchRequest(BaseModel):
    user_id: str
    top_k: int = 10

# ------------------ APIs ------------------

@app.post("/user/register")
def register_user(data: RegisterRequest):
    if data.user_id in user_vectors:
        raise HTTPException(status_code=400, detail="User already exists")

    vector = create_weighted_vector(data.responses)
    add_user_vector(data.user_id, vector)

    return {"status": "success", "message": "User vector stored"}

@app.post("/matches")
def find_matches(data: MatchRequest):
    if data.user_id not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    query_vector = user_vectors[data.user_id]
    results = search(query_vector, data.top_k + 1)

    # remove self-match
    matches = [
        {"user_id": uid, "score": score}
        for uid, score in results
        if uid != data.user_id
    ][:data.top_k]

    return {"matches": matches}
