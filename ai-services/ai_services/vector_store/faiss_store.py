"""FAISS-backed vector store with JSON metadata persistence.

Each embedding backend gets its own store (different dimensions), persisted under
`{path}/index.faiss` and `{path}/meta.json`. Vectors are L2-normalized so the
`IndexFlatIP` inner product is equivalent to cosine similarity.
"""

import json
import os

import faiss
import numpy as np

from .base import VectorMatch, VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int, path: str) -> None:
        self.dim = dim
        self.path = path
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.metadata: dict[str, dict] = {}
        self._vectors: list[np.ndarray] = []
        self._load()

    def add(self, item_id: str, vector: np.ndarray, metadata: dict) -> None:
        normalized = _normalize(vector).astype("float32")
        self.index.add(normalized.reshape(1, -1))
        self.ids.append(item_id)
        self.metadata[item_id] = metadata
        self._vectors.append(normalized)

    def search(self, vector: np.ndarray, k: int = 5) -> list[VectorMatch]:
        if self.index.ntotal == 0:
            return []
        query = _normalize(vector).astype("float32").reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item_id = self.ids[idx]
            matches.append(VectorMatch(id=item_id, score=float(score), metadata=self.metadata[item_id]))
        return matches

    def project_2d(self) -> list[dict]:
        if not self.ids:
            return []
        if len(self.ids) == 1:
            return [{"id": self.ids[0], "x": 0.0, "y": 0.0, **self.metadata[self.ids[0]]}]

        matrix = np.stack(self._vectors)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ components[:2].T

        max_abs = np.abs(coords).max()
        if max_abs > 0:
            coords = coords / max_abs

        return [
            {"id": item_id, "x": float(point[0]), "y": float(point[1]), **self.metadata[item_id]}
            for item_id, point in zip(self.ids, coords)
        ]

    def persist(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.path, "index.faiss"))
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "ids": self.ids,
                    "metadata": self.metadata,
                    "vectors": [vector.tolist() for vector in self._vectors],
                },
                handle,
            )

    def _load(self) -> None:
        meta_path = os.path.join(self.path, "meta.json")
        index_path = os.path.join(self.path, "index.faiss")
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            return

        self.index = faiss.read_index(index_path)
        with open(meta_path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        self._vectors = [np.array(vector, dtype=np.float32) for vector in data["vectors"]]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
