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

        neighbors_df = pd.read_parquet(neighbors_path)
        self._neighbors_df = neighbors_df.set_index("source_game_id").sort_index()

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
        try:
            rows = self._neighbors_df.loc[[game_id]]
        except KeyError:
            return []

        rows = rows.sort_values("score", ascending=False).head(top_k)
        results = []
        for row in rows.itertuples():
            game = self._games_by_id.get(row.neighbor_game_id)
            if game is None:
                continue
            results.append(
                Recommendation(
                    game=game,
                    score=row.score,
                    reasons=row.reasons.split("|"),
                    shared_genres=row.shared_genres.split("|") if row.shared_genres else [],
                )
            )
        return results



@lru_cache
def get_recommender() -> Recommender:
    return Recommender(settings.games_path, settings.neighbors_path)
