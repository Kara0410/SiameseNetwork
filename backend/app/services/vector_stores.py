"""Per-model FAISS vector store instances, cached for the process lifetime.

Each embedding backend has its own dimensionality, so each gets its own store
persisted under `{VECTOR_STORE_DIR}/{model_name}/`.
"""

import os
from functools import lru_cache

from ai_services.embeddings import get_model
from ai_services.vector_store import FaissVectorStore

from ..core.config import get_settings


@lru_cache
def get_vector_store(model_name: str) -> FaissVectorStore:
    settings = get_settings()
    model = get_model(model_name)
    path = os.path.join(settings.vector_store_dir, model_name)
    return FaissVectorStore(dim=model.info.dimension, path=path)
