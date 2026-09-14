from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    games_path: Path = BASE_DIR / "data" / "processed" / "games.parquet"
    neighbors_path: Path = BASE_DIR / "data" / "models" / "hybrid_neighbors.json"
    default_top_k: int = 6


settings = Settings()
