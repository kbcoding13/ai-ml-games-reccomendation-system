"""Download the raw Kaggle datasets used to train the hybrid recommender.

Requires a Kaggle account and API token configured for kagglehub
(https://github.com/Kaggle/kagglehub#authenticate). Run once before the
build_*.py scripts.
"""

import shutil
from pathlib import Path

import kagglehub

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _copy_dataset(handle: str, wanted_files: list[str]) -> None:
    src_dir = Path(kagglehub.dataset_download(handle))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in wanted_files:
        src = next(src_dir.rglob(name), None)
        if src is None:
            print(f"  ! could not find {name} in {handle}, skipping")
            continue
        dest = RAW_DIR / name
        shutil.copy(src, dest)
        print(f"  -> {dest}")


def main() -> None:
    print("Downloading game metadata (nikdavis/steam-store-games)...")
    _copy_dataset("nikdavis/steam-store-games", ["steam.csv"])

    print("Downloading user playtime data (tamber/steam-video-games)...")
    _copy_dataset("tamber/steam-video-games", ["steam-200k.csv"])

    print("Done. Raw files are in data/raw/.")


if __name__ == "__main__":
    main()
