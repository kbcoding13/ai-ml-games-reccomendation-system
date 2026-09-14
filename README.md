# Game Recommendation System

A hybrid game recommender: pick a game you like, get similar ones back with a
plain-English reason for each suggestion.

Inspired by [inboxpraveen/Movie-Recommendation-System](https://github.com/inboxpraveen/Movie-Recommendation-System)
(content-based movie recommendations via TF-IDF + SVD, precomputed neighbor
lists for scale). This project reimplements the idea from scratch for a
different domain (video games instead of movies) with its own stack and adds
a **hybrid** scoring model plus per-recommendation explanations, which the
original does not have.

## How it works

1. **Content-based model** — TF-IDF over each game's genres/categories/tags,
   reduced with TruncatedSVD, top-K neighbors precomputed by cosine
   similarity ("games with similar genres/tags").
2. **Collaborative-filtering model** — ALS matrix factorization on implicit
   feedback (hours played), top-K neighbors from the learned item factors
   ("games played by similar players").
3. **Hybrid index** — the two neighbor lists are merged with a weighted
   score, so recommendations aren't purely genre-matching (which can be
   repetitive) or purely popularity-driven (which struggles on niche
   titles). Every recommendation also carries its shared genres and a
   `reasons` list explaining the match.
4. Like the original project, only the top-K neighbors per game are stored
   (not a full N×N similarity matrix), so this scales to a full Steam
   catalog without blowing up memory.

## Stack

- **Backend**: FastAPI + Pydantic
- **ML/training**: scikit-learn (TF-IDF, SVD, nearest neighbors), implicit
  (ALS collaborative filtering), pandas/numpy, Parquet for storage
- **Frontend**: plain HTML/CSS/JS (no framework needed for a search box +
  results grid)
- **Deployment**: Docker

## Running locally (demo data, no setup required)

A small hand-curated dataset (20 well-known games) ships in `data/demo/` so
the app works immediately:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and search for a game (e.g. "Hades").

## Training on the full dataset

To go from the 20-game demo to a real Steam-scale model:

1. Set up Kaggle API credentials for `kagglehub` (https://github.com/Kaggle/kagglehub#authenticate).
2. Install training dependencies: `pip install -r requirements-train.txt`
3. Run the pipeline in order:
   ```bash
   python training/download_data.py
   python training/build_content_model.py
   python training/build_collab_model.py
   python training/build_hybrid_index.py
   ```
   This downloads the Steam game-metadata and player-playtime datasets from
   Kaggle, builds both neighbor models, and merges them into
   `data/processed/games.parquet` and `data/models/hybrid_neighbors.json`.
4. Copy `.env.example` to `.env` and point `GAMES_PATH` / `NEIGHBORS_PATH` at
   those `data/processed` / `data/models` files instead of the demo dataset.
5. Restart the app.

## API

- `GET /api/search?q=<query>` — fuzzy-ish substring search by name
- `GET /api/games/{game_id}` — game details
- `GET /api/games/{game_id}/recommendations?top_k=6` — hybrid recommendations
  with `score`, `reasons`, and `shared_genres` per result

## Docker

```bash
docker build -t game-recs .
docker run -p 8000:8000 game-recs
```

Ships with the demo dataset baked in; mount/replace `data/demo` (or set the
env vars) with a fully trained model for production use.
