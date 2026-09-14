"""Build the content-based half of the hybrid recommender.

Vectorizes each game's genres/categories/tags with TF-IDF, reduces with
TruncatedSVD, then precomputes the top-K nearest neighbors per game by
cosine similarity - mirroring the "store neighbors, not a full similarity
matrix" trick from the original movie project so this scales past a few
thousand titles.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

TOP_K = 20
SVD_COMPONENTS = 200


def _tag_text(row: pd.Series) -> str:
    parts = []
    for col in ("genres", "categories", "steamspy_tags"):
        val = row.get(col)
        if isinstance(val, str) and val:
            parts.append(val.replace(";", " "))
    return " ".join(parts)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / "steam.csv")
    df = df.dropna(subset=["name"]).reset_index(drop=True)
    df["tag_text"] = df.apply(_tag_text, axis=1)

    games = df[
        ["appid", "name", "genres", "categories", "steamspy_tags", "positive_ratings", "negative_ratings"]
    ].rename(columns={"appid": "game_id"})
    games.to_parquet(PROCESSED_DIR / "games.parquet", index=False)

    print(f"Vectorizing tag text for {len(df)} games...")
    tfidf = TfidfVectorizer(min_df=2, max_features=20_000)
    tfidf_matrix = tfidf.fit_transform(df["tag_text"])

    n_components = min(SVD_COMPONENTS, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)

    print(f"Finding top-{TOP_K} content neighbors per game...")
    nn = NearestNeighbors(n_neighbors=TOP_K + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    game_ids = df["appid"].to_numpy()
    genre_lists = df["genres"].fillna("").str.split(";").apply(lambda xs: {x.strip() for x in xs if x})

    neighbors = {}
    for row_idx, game_id in enumerate(game_ids):
        row_neighbors = []
        for dist, neighbor_idx in zip(distances[row_idx], indices[row_idx]):
            if neighbor_idx == row_idx:
                continue
            shared_genres = sorted(genre_lists[row_idx] & genre_lists[neighbor_idx])
            row_neighbors.append(
                {
                    "game_id": int(game_ids[neighbor_idx]),
                    "score": float(1 - dist),
                    "shared_genres": shared_genres[:3],
                }
            )
            if len(row_neighbors) == TOP_K:
                break
        neighbors[int(game_id)] = row_neighbors

    with open(MODELS_DIR / "content_neighbors.json", "w") as f:
        json.dump(neighbors, f)

    np.save(MODELS_DIR / "content_embeddings.npy", embeddings)
    print("Saved content_neighbors.json and content_embeddings.npy")


if __name__ == "__main__":
    main()
