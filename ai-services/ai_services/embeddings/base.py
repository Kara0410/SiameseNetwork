"""Shared interface every embedding backend implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Static metadata describing an embedding backend, exposed via `GET /models`."""

    name: str
    display_name: str
    dimension: int
    description: str
    # "gradcam" | "attention" | "heuristic" | "none"
    explainability: str


class EmbeddingModel(ABC):
    """A model that turns an image into a unit-normalized embedding vector."""

    info: EmbeddingModelInfo

    @abstractmethod
    def embed(self, image: Image.Image) -> np.ndarray:
        """Return a unit-normalized 1D embedding vector for `image`."""

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Embed multiple images, stacked into a `(N, dim)` array."""
        return np.stack([self.embed(image) for image in images])

    def explain(self, image: Image.Image) -> Optional[np.ndarray]:
        """Return a 2D heatmap in `[0, 1]` highlighting which regions drove the
        embedding, or `None` if this backend has no explainability support."""
        return None
