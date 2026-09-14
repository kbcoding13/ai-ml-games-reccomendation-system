from pydantic import BaseModel


class Game(BaseModel):
    game_id: int
    name: str
    genres: list[str]


class Recommendation(BaseModel):
    game: Game
    score: float
    reasons: list[str]
    shared_genres: list[str]


class SearchResult(BaseModel):
    games: list[Game]
