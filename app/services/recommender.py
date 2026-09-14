import json
from functools import lru_cache

import pandas as pd

from app.core.config import settings
from app.schemas import Game, Recommendation


class Recommender:
    def __init__(self, games_path, neighbors_path):
        games_df = pd.read_parquet(games_path)
        games_df["genres"] = games_df["genres"].fillna("").apply(
            lambda s: [g for g in s.split(";") if g]
        )
        self._games_by_id: dict[int, Game] = {
            int(row.game_id): Game(game_id=int(row.game_id), name=row.name, genres=row.genres)
            for row in games_df.itertuples()
        }
        self._name_lower_index = [
            (game.name.lower(), game) for game in self._games_by_id.values()
        ]

        with open(neighbors_path) as f:
            raw_neighbors = json.load(f)
        self._neighbors: dict[int, list[dict]] = {
            int(game_id): neighbors for game_id, neighbors in raw_neighbors.items()
        }

    def search(self, query: str, limit: int = 10) -> list[Game]:
        query = query.strip().lower()
        if not query:
            return []
        matches = [game for name, game in self._name_lower_index if query in name]
        matches.sort(key=lambda g: (not g.name.lower().startswith(query), g.name))
        return matches[:limit]

    def get_game(self, game_id: int) -> Game | None:
        return self._games_by_id.get(game_id)

    def recommend(self, game_id: int, top_k: int) -> list[Recommendation]:
        neighbors = self._neighbors.get(game_id, [])[:top_k]
        results = []
        for neighbor in neighbors:
            game = self._games_by_id.get(neighbor["game_id"])
            if game is None:
                continue
            results.append(
                Recommendation(
                    game=game,
                    score=neighbor["score"],
                    reasons=neighbor["reasons"],
                    shared_genres=neighbor.get("shared_genres", []),
                )
            )
        return results


@lru_cache
def get_recommender() -> Recommender:
    return Recommender(settings.games_path, settings.neighbors_path)
