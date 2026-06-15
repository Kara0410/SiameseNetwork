"""Vision Transformer (ViT) embedding backend.

Uses a HuggingFace `ViTModel` for embeddings via the pooled `[CLS]` token, with
attention-rollout explainability. Weights are downloaded lazily on first use and
cached under `ai_services.config.MODEL_CACHE_DIR`.
"""

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from .. import config
from ..explainability.attention_map import attention_rollout
from .base import EmbeddingModel, EmbeddingModelInfo

DEFAULT_MODEL_ID = "google/vit-base-patch16-224-in21k"


class ViTEmbeddingModel(EmbeddingModel):
    info = EmbeddingModelInfo(
        name="vit",
        display_name="ViT-B/16",
        dimension=768,
        description="Vision Transformer encoder - pooled CLS-token embeddings with "
        "attention-rollout explainability.",
        explainability="attention",
    )

    _model: ViTModel | None = None
    _processor: ViTImageProcessor | None = None

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def _load(self) -> tuple[ViTModel, ViTImageProcessor]:
        if ViTEmbeddingModel._model is None:
            ViTEmbeddingModel._model = (
                ViTModel.from_pretrained(self.model_id, cache_dir=config.MODEL_CACHE_DIR).to(config.DEVICE).eval()
            )
            ViTEmbeddingModel._processor = ViTImageProcessor.from_pretrained(
                self.model_id, cache_dir=config.MODEL_CACHE_DIR
            )
        return ViTEmbeddingModel._model, ViTEmbeddingModel._processor

    def embed(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.pooler_output[0].cpu().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def explain(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        return attention_rollout(outputs.attentions)
