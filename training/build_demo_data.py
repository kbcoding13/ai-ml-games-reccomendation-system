"""Generate a small hand-curated demo dataset so the app runs out of the box
without a Kaggle account. Writes the exact same files the real pipeline
(download_data.py -> build_content_model.py -> build_collab_model.py ->
build_hybrid_index.py) produces, so `uvicorn app.main:app` works immediately
and can later be swapped for the full-scale trained artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"

DEMO_GAMES = [
    (1, "Stardew Valley", "Indie;RPG;Simulation", "Farming Sim;Relaxing;Singleplayer", 95, 2),
    (2, "Terraria", "Action;Adventure;Indie;RPG", "Sandbox;Crafting;2D", 96, 3),
    (3, "Hades", "Action;Indie;RPG", "Roguelike;Great Soundtrack;Singleplayer", 98, 1),
    (4, "Dead Cells", "Action;Indie;RPG", "Roguelike;Metroidvania;2D", 94, 2),
    (5, "Hollow Knight", "Action;Adventure;Indie", "Metroidvania;Atmospheric;Difficult", 97, 1),
    (6, "Celeste", "Action;Indie;Platformer", "Difficult;Pixel Graphics;Story Rich", 96, 1),
    (7, "Dark Souls III", "Action;RPG", "Souls-like;Difficult;Atmospheric", 93, 4),
    (8, "Elden Ring", "Action;RPG", "Souls-like;Open World;Atmospheric", 95, 5),
    (9, "Portal 2", "Action;Puzzle", "Puzzle;Singleplayer;Co-op", 98, 1),
    (10, "The Witness", "Adventure;Indie;Puzzle", "Puzzle;Exploration;Atmospheric", 90, 1),
    (11, "Slay the Spire", "Indie;RPG;Strategy", "Roguelike;Deckbuilding;Singleplayer", 97, 2),
    (12, "Balatro", "Indie;Strategy", "Deckbuilding;Roguelike;Addictive", 97, 1),
    (13, "Civilization VI", "Strategy", "4X;Turn-Based Strategy;Multiplayer", 88, 6),
    (14, "Into the Breach", "Indie;Strategy", "Turn-Based Strategy;Roguelike;Singleplayer", 93, 1),
    (15, "Cities: Skylines", "Simulation;Strategy", "City Builder;Sandbox;Singleplayer", 89, 7),
    (16, "Factorio", "Automation;Indie;Simulation", "Sandbox;Crafting;Automation", 97, 3),
    (17, "Portal", "Action;Puzzle", "Puzzle;Singleplayer;Atmospheric", 96, 1),
    (18, "Ori and the Blind Forest", "Action;Adventure;Indie", "Metroidvania;Atmospheric;Platformer", 95, 1),
    (19, "Risk of Rain 2", "Action;Indie;RPG", "Roguelike;Co-op;Multiplayer", 92, 2),
    (20, "Divinity: Original Sin 2", "Adventure;RPG;Strategy", "Turn-Based Strategy;Co-op;Story Rich", 96, 8),
]

TOP_K = 8


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        DEMO_GAMES,
        columns=["game_id", "name", "genres", "steamspy_tags", "positive_ratings", "negative_ratings"],
    )
    df["categories"] = ""
    df.to_parquet(DEMO_DIR / "games.parquet", index=False)

    tag_text = (df["genres"] + " " + df["steamspy_tags"]).str.replace(";", " ", regex=False)
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(tag_text)
    sims = cosine_similarity(matrix)

    genre_lists = df["genres"].str.split(";").apply(set)

    hybrid = {}
    for i, game_id in enumerate(df["game_id"]):
        order = np.argsort(-sims[i])
        neighbors = []
        for j in order:
            if j == i:
                continue
            shared = sorted(genre_lists[i] & genre_lists[j])
            neighbors.append(
                {
                    "game_id": int(df["game_id"][j]),
                    "score": round(float(sims[i][j]), 4),
                    "reasons": ["similar genres/tags"],
                    "shared_genres": shared[:3],
                }
            )
            if len(neighbors) == TOP_K:
                break
        hybrid[str(int(game_id))] = neighbors

    with open(DEMO_DIR / "hybrid_neighbors.json", "w") as f:
        json.dump(hybrid, f)

    print(f"Wrote demo dataset: {len(df)} games -> {DEMO_DIR / 'games.parquet'}")
    print(f"Wrote demo neighbors -> {DEMO_DIR / 'hybrid_neighbors.json'}")


if __name__ == "__main__":
    main()
