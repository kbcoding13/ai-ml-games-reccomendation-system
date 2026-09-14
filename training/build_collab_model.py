"""Build the collaborative-filtering half of the hybrid recommender.

Uses implicit feedback (hours played, from the Steam-200k dataset) as a
confidence signal, trains an ALS matrix factorization model, and
precomputes top-K item-item neighbors from the learned item factors -
i.e. "players who played X also played Y", independent of genre/tag text.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

TOP_K = 20


def main() -> None:
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    name_to_id = dict(zip(games["name"].str.lower(), games["game_id"]))

    interactions = pd.read_csv(
        RAW_DIR / "steam-200k.csv",
        header=None,
        names=["user_id", "game_name", "behavior", "value", "_unused"],
    )
    play = interactions[interactions["behavior"] == "play"].copy()
    play["game_id"] = play["game_name"].str.lower().map(name_to_id)
    play = play.dropna(subset=["game_id"])
    play["game_id"] = play["game_id"].astype(int)

    play["user_idx"] = play["user_id"].astype("category").cat.codes
    play["game_idx"] = play["game_id"].astype("category").cat.codes
    game_idx_to_id = dict(enumerate(play["game_id"].astype("category").cat.categories))

    confidence = 1 + np.log1p(play["value"].astype(float))
    matrix = coo_matrix(
        (confidence, (play["user_idx"], play["game_idx"])),
        shape=(play["user_idx"].nunique(), play["game_idx"].nunique()),
    ).tocsr()

    print(f"Training ALS on {matrix.shape[0]} users x {matrix.shape[1]} games...")
    model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=42)
    model.fit(matrix)

    item_factors = model.item_factors
    item_factors = item_factors / np.linalg.norm(item_factors, axis=1, keepdims=True).clip(min=1e-9)

    print(f"Finding top-{TOP_K} collaborative neighbors per game...")
    nn = NearestNeighbors(n_neighbors=min(TOP_K + 1, item_factors.shape[0]), metric="cosine")
    nn.fit(item_factors)
    distances, indices = nn.kneighbors(item_factors)

    neighbors = {}
    for row_idx in range(item_factors.shape[0]):
        game_id = game_idx_to_id[row_idx]
        row_neighbors = []
        for dist, neighbor_idx in zip(distances[row_idx], indices[row_idx]):
            if neighbor_idx == row_idx:
                continue
            row_neighbors.append(
                {"game_id": int(game_idx_to_id[neighbor_idx]), "score": float(1 - dist)}
            )
        neighbors[int(game_id)] = row_neighbors[:TOP_K]

    with open(MODELS_DIR / "collab_neighbors.json", "w") as f:
        json.dump(neighbors, f)

    print("Saved collab_neighbors.json")


if __name__ == "__main__":
    main()
