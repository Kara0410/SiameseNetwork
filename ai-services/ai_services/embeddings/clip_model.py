"""CLIP-based embedding backend.

Uses OpenAI's CLIP vision encoder via HuggingFace `transformers` for general-purpose
semantic image embeddings. Weights are downloaded lazily on first use and cached under
`ai_services.config.MODEL_CACHE_DIR`.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .. import config
from ..explainability.attention_map import attention_rollout
from .base import EmbeddingModel, EmbeddingModelInfo

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class ClipEmbeddingModel(EmbeddingModel):
    info = EmbeddingModelInfo(
        name="clip",
        display_name="CLIP ViT-B/32",
        dimension=512,
        description="OpenAI CLIP vision encoder - general-purpose semantic embeddings "
        "with attention-rollout explainability.",
        explainability="attention",
    )

    _model: CLIPModel | None = None
    _processor: CLIPProcessor | None = None

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id

    def _load(self) -> tuple[CLIPModel, CLIPProcessor]:
        if ClipEmbeddingModel._model is None:
            ClipEmbeddingModel._model = (
                CLIPModel.from_pretrained(self.model_id, cache_dir=config.MODEL_CACHE_DIR).to(config.DEVICE).eval()
            )
            ClipEmbeddingModel._processor = CLIPProcessor.from_pretrained(
                self.model_id, cache_dir=config.MODEL_CACHE_DIR
            )
        return ClipEmbeddingModel._model, ClipEmbeddingModel._processor

    def embed(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vector = features[0].cpu().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def explain(self, image: Image.Image) -> np.ndarray:
        model, processor = self._load()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
        return attention_rollout(vision_outputs.attentions)
