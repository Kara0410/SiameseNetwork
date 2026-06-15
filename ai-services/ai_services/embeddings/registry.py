"""Lazily-imported registry of embedding backends.

Each backend module is only imported (and therefore only imports torch/transformers/
torchvision) when it is actually requested, so selecting `dummy` never touches the
heavy ML stack.
"""

import importlib

from .base import EmbeddingModel, EmbeddingModelInfo

# name -> (module path, class name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "dummy": ("ai_services.embeddings.dummy_model", "DummyEmbeddingModel"),
    "clip": ("ai_services.embeddings.clip_model", "ClipEmbeddingModel"),
    "vit": ("ai_services.embeddings.vit_model", "ViTEmbeddingModel"),
    "efficientnet": ("ai_services.embeddings.efficientnet_model", "EfficientNetEmbeddingModel"),
    "siamese": ("ai_services.embeddings.siamese_model", "SiameseEmbeddingModel"),
}

_INSTANCES: dict[str, EmbeddingModel] = {}


def _load_class(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown embedding model '{name}'. Available: {sorted(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_model(name: str) -> EmbeddingModel:
    """Return a cached instance of the embedding backend registered as `name`."""
    if name not in _INSTANCES:
        _INSTANCES[name] = _load_class(name)()
    return _INSTANCES[name]


def list_models() -> list[EmbeddingModelInfo]:
    """Return static metadata for every registered embedding backend."""
    return [_load_class(name).info for name in _REGISTRY]


def available_models() -> list[str]:
    """Return the registry keys, without importing any backend module."""
    return sorted(_REGISTRY)
