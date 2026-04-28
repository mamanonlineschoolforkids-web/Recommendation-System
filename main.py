from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel
import os

# =========================
# إعداد التطبيق
# =========================
app = FastAPI(title="Recommendation System API 🚀")

# =========================
# تحميل البيانات مرة واحدة
# =========================
try:
    content = pd.read_csv("content_edu_2000.csv")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading data: {e}")

# =========================
# تحديد البورت (مهم لـ Railway)
# =========================
PORT = int(os.environ.get("PORT", 8000))

# =========================
# Request Model (POST)
# =========================
class RecommendationRequest(BaseModel):
    content_id: int | None = None
    title: str | None = None
    top_n: int = 5

# =========================
# Functions
# =========================

def get_index_by_id(content_id):
    matches = content[content['content_id'] == content_id]
    if matches.empty:
        return None
    return matches.index[0]


def get_index_by_title(title):
    matches = content[content['title'].str.lower() == title.lower()]
    if matches.empty:
        return None
    return matches.index[0]


def generate_recommendations(idx, top_n=5):
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    results = content.iloc[similar_indices][['content_id', 'title']]
    return results.to_dict(orient="records")

# =========================
# Endpoints
# =========================

@app.get("/")
def home():
    return {"message": "🔥 Recommendation API is running"}

# 🔹 باستخدام ID
@app.get("/recommend/id/{content_id}")
def recommend_by_id(content_id: int, top_n: int = 5):
    idx = get_index_by_id(content_id)

    if idx is None:
        raise HTTPException(status_code=404, detail="content_id not found")

    recs = generate_recommendations(idx, top_n)

    return {
        "input": content_id,
        "type": "content_id",
        "recommendations": recs
    }

# 🔹 باستخدام Title
@app.get("/recommend/title")
def recommend_by_title(title: str, top_n: int = 5):
    idx = get_index_by_title(title)

    if idx is None:
        raise HTTPException(status_code=404, detail="title not found")

    recs = generate_recommendations(idx, top_n)

    return {
        "input": title,
        "type": "title",
        "recommendations": recs
    }

# 🔹 POST Endpoint
@app.post("/recommend")
def recommend_post(request: RecommendationRequest):

    if request.content_id is not None:
        idx = get_index_by_id(request.content_id)
        input_type = "content_id"
        input_value = request.content_id

    elif request.title is not None:
        idx = get_index_by_title(request.title)
        input_type = "title"
        input_value = request.title

    else:
        raise HTTPException(status_code=400, detail="Provide content_id or title")

    if idx is None:
        raise HTTPException(status_code=404, detail="Item not found")

    recs = generate_recommendations(idx, request.top_n)

    return {
        "input": input_value,
        "type": input_type,
        "recommendations": recs
    }