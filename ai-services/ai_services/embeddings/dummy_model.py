"""Deterministic, dependency-light embedding backend used for local dev, CI, and tests.

Produces a real (if low-quality) embedding from pixel content via a fixed random
projection, so similarity scores and explainability heatmaps behave sensibly without
downloading any model weights.
"""

import numpy as np
from PIL import Image, ImageFilter

from .base import EmbeddingModel, EmbeddingModelInfo

_PROJECTION_SEED = 42
_INPUT_SIZE = 32


class DummyEmbeddingModel(EmbeddingModel):
    info = EmbeddingModelInfo(
        name="dummy",
        display_name="Dummy (offline dev)",
        dimension=64,
        description="Deterministic hash-based embedding for local development, CI, and "
        "tests - no model downloads required.",
        explainability="heuristic",
    )

    def __init__(self) -> None:
        rng = np.random.default_rng(_PROJECTION_SEED)
        self._projection = rng.standard_normal((_INPUT_SIZE * _INPUT_SIZE, self.info.dimension)).astype(
            np.float32
        )

    def embed(self, image: Image.Image) -> np.ndarray:
        gray = image.convert("L").resize((_INPUT_SIZE, _INPUT_SIZE))
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
        vector = pixels.flatten() @ self._projection
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def explain(self, image: Image.Image) -> np.ndarray:
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        heatmap = np.asarray(edges, dtype=np.float32)
        heatmap -= heatmap.min()
        max_value = heatmap.max()
        if max_value > 0:
            heatmap /= max_value
        return heatmap
