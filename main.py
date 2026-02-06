from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from faiss_store_match import add_user_vector, search, male_rollno_vectors,female_rollno_vectors,male_id_to_user,female_id_to_user
from vectorization import create_weighted_vector

app = FastAPI(title="Dating App Matching API")
user_genders = {}

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
    if ((data.rollno in male_rollno_vectors) or (data.rollno in female_rollno_vectors)):
        raise HTTPException(status_code=400, detail="User already exists")
    user_genders[data.rollno] = data.responses['q4']
    vector = create_weighted_vector(data.responses)
    add_user_vector(data.rollno, vector,user_genders[data.rollno])

    return {"status": "success", "message": "User vector stored"}

@app.post("/matches")
def find_matches(data: MatchRequest):
    if data.rollno not in male_rollno_vectors or data.rollno not in female_rollno_vectors:
        raise HTTPException(status_code=404, detail="User not found")
    gender = user_genders[data.rollno]
    if(gender == 'male'):
        query_vector = male_rollno_vectors[data.rollno]
    elif(gender == 'female'):
        query_vector = female_rollno_vectors[data.rollno]
    results = search(query_vector, data.top_k + 1,gender)

    # remove self-match
    matches = [
        {"rollno": uid, "score": score}
        for uid, score in results
        if uid != data.rollno
    ][:data.top_k]

    return {"matches": matches}
