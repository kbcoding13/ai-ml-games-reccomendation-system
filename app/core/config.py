from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Defaults point at the small, committed demo dataset so the app runs
    # out of the box. Point these at data/processed and data/models
    # (via .env) after running the full training pipeline for the
    # full-scale (930k+ / Steam-catalog) model.
    games_path: Path = BASE_DIR / "data" / "demo" / "games.parquet"
    neighbors_path: Path = BASE_DIR / "data" / "demo" / "hybrid_neighbors.json"
    default_top_k: int = 6


settings = Settings()
