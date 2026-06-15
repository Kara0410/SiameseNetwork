"""Seed a model's FAISS vector store with demo identities.

Without this, the dashboard's embedding map and vector-search panel are empty until
real `/verify` calls have been made. This script embeds a handful of synthetic images
and writes them into the same vector store the backend reads from, so the panels have
content immediately after `docker compose up` or `uvicorn app.main:app`.

Usage:
    python scripts/seed_vector_store.py                  # seeds the `dummy` backend
    python scripts/seed_vector_store.py --model clip --count 12

The store path mirrors the backend's `VISIONIQ_VECTOR_STORE_DIR` setting
(default `data/vector_store`, resolved relative to `backend/`).
"""

import argparse
import os

import numpy as np
from PIL import Image

from ai_services.embeddings.registry import get_model
from ai_services.vector_store.faiss_store import FaissVectorStore

_VERDICTS = ["verified", "verified", "verified", "review", "blocked"]


def main(model_name: str, count: int, vector_store_dir: str) -> None:
    model = get_model(model_name)
    store_path = os.path.join(vector_store_dir, model_name)
    store = FaissVectorStore(dim=model.info.dimension, path=store_path)

    rng = np.random.default_rng(42)
    for i in range(count):
        image = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        vector = model.embed(image)
        verdict = _VERDICTS[i % len(_VERDICTS)]
        item_id = f"seed_{i:03d}"
        similarity = float(np.clip(0.6 + rng.random() * 0.4, 0.0, 1.0))
        store.add(item_id, vector, {"label": item_id, "verdict": verdict, "similarity": similarity})

    store.persist()
    print(f"Seeded {count} vectors for '{model_name}' into {store_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dummy", help="Embedding backend to seed (default: dummy)")
    parser.add_argument("--count", type=int, default=24, help="Number of synthetic vectors to add")
    parser.add_argument(
        "--vector-store-dir",
        default=os.environ.get("VISIONIQ_VECTOR_STORE_DIR", "backend/data/vector_store"),
        help="Base vector store directory (default: $VISIONIQ_VECTOR_STORE_DIR or backend/data/vector_store)",
    )
    args = parser.parse_args()
    main(args.model, args.count, args.vector_store_dir)
