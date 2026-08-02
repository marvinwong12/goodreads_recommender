# app/main.py
"""
FastAPI wrapper around RecommenderService: a user searches for and picks
books they've enjoyed, then gets live recommendations through the full
two-stage pipeline (fused + co-occurrence retrieval, XGBoost reranking).
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference.recommend import RecommenderService

service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = RecommenderService()
    yield


app = FastAPI(title="Goodreads Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    book_ids: list[str]
    user_avg_rating: float | None = None
    top_k: int = 10


@app.get("/api/search")
def search(q: str, limit: int = 10):
    return service.search_books(q, limit=limit)


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    if not req.book_ids:
        raise HTTPException(status_code=400, detail="Provide at least one book_id.")
    try:
        return service.recommend(req.book_ids, user_avg_rating=req.user_avg_rating, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


app.mount("/static", StaticFiles(directory=CURRENT_FILE.parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(CURRENT_FILE.parent / "static" / "index.html")
