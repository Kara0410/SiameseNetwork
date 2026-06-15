"""Shared interface for embedding vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VectorMatch:
    """A single nearest-neighbor result from `VectorStore.search`."""

    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Stores embedding vectors with metadata and supports similarity search."""

    @abstractmethod
    def add(self, item_id: str, vector: np.ndarray, metadata: dict) -> None:
        """Add (or overwrite) `item_id` with `vector` and `metadata`."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int = 5) -> list[VectorMatch]:
        """Return the `k` nearest neighbors of `vector`, ranked by cosine similarity."""

    @abstractmethod
    def project_2d(self) -> list[dict]:
        """Project all stored vectors into 2D (via PCA) for embedding-map visualizations.

        Returns a list of `{"id": ..., "x": ..., "y": ..., **metadata}` dicts.
        """

    @abstractmethod
    def persist(self) -> None:
        """Write the index and metadata to disk."""
