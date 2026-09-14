from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.schemas import Game, Recommendation, SearchResult
from app.services.recommender import get_recommender

router = APIRouter(prefix="/api")


@router.get("/search", response_model=SearchResult)
def search_games(q: str = Query(..., min_length=1)):
    recommender = get_recommender()
    return SearchResult(games=recommender.search(q))


@router.get("/games/{game_id}", response_model=Game)
def get_game(game_id: int):
    recommender = get_recommender()
    game = recommender.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@router.get("/games/{game_id}/recommendations", response_model=list[Recommendation])
def recommend_games(game_id: int, top_k: int = Query(default=settings.default_top_k, ge=1, le=20)):
    recommender = get_recommender()
    if recommender.get_game(game_id) is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return recommender.recommend(game_id, top_k)
