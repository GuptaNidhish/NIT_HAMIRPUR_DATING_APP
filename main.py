from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from faiss_store import add_user_vector, search, user_vectors
from vectorization import create_weighted_vector

app = FastAPI(title="Dating App Matching API")

# ------------------ Schemas ------------------

class RegisterRequest(BaseModel):
    rollno: str
    responses: Dict[str, str]

class MatchRequest(BaseModel):
    rollno: str
    top_k: int = 10

# ------------------ APIs ------------------

@app.post("/user/register")
def register_user(data: RegisterRequest):
    if data.rollno in rollno_vectors:
        raise HTTPException(status_code=400, detail="User already exists")

    vector = create_weighted_vector(data.responses)
    add_user_vector(data.user_id, vector)

    return {"status": "success", "message": "User vector stored"}

@app.post("/matches")
def find_matches(data: MatchRequest):
    if data.user_id not in user_vectors:
        raise HTTPException(status_code=404, detail="User not found")

    query_vector = rollno_vectors[data.rollno]
    results = search(query_vector, data.top_k + 1)

    # remove self-match
    matches = [
        {"rollno": uid, "score": score}
        for uid, score in results
        if uid != data.roolno
    ][:data.top_k]

    return {"matches": matches}
