"""Merge the content-based and collaborative neighbor lists into one
hybrid index, with a human-readable reason attached to every recommendation.

Games missing from the collaborative model (no play data) fall back to
content-only scoring - this is what keeps the system usable for the long
tail of niche/indie titles that never got a large player base.
"""

import json
from pathlib import Path

CONTENT_WEIGHT = 0.6
COLLAB_WEIGHT = 0.4
TOP_K = 10

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def main() -> None:
    with open(MODELS_DIR / "content_neighbors.json") as f:
        content = json.load(f)
    with open(MODELS_DIR / "collab_neighbors.json") as f:
        collab = json.load(f)

    hybrid = {}
    for game_id, content_neighbors in content.items():
        collab_neighbors = {n["game_id"]: n["score"] for n in collab.get(game_id, [])}
        content_by_id = {n["game_id"]: n for n in content_neighbors}

        candidate_ids = set(content_by_id) | set(collab_neighbors)
        scored = []
        for cand_id in candidate_ids:
            c_score = content_by_id.get(cand_id, {}).get("score", 0.0)
            v_score = collab_neighbors.get(cand_id, 0.0)

            has_content = cand_id in content_by_id
            has_collab = cand_id in collab_neighbors
            if has_content and has_collab:
                score = CONTENT_WEIGHT * c_score + COLLAB_WEIGHT * v_score
                reasons = ["similar genres/tags", "played by similar players"]
            elif has_content:
                score = c_score
                reasons = ["similar genres/tags"]
            else:
                score = v_score
                reasons = ["played by similar players"]

            shared_genres = content_by_id.get(cand_id, {}).get("shared_genres", [])
            scored.append(
                {
                    "game_id": cand_id,
                    "score": round(score, 4),
                    "reasons": reasons,
                    "shared_genres": shared_genres,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        hybrid[game_id] = scored[:TOP_K]

    with open(MODELS_DIR / "hybrid_neighbors.json", "w") as f:
        json.dump(hybrid, f)

    print(f"Saved hybrid_neighbors.json with {len(hybrid)} games")


if __name__ == "__main__":
    main()
