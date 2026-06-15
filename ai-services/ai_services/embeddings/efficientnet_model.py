"""EfficientNet-B0 embedding backend.

Uses torchvision's ImageNet-pretrained EfficientNet-B0 with its classifier head
removed, exposing the 1280-d pooled feature vector as the embedding. Grad-CAM hooks
the final convolutional block for explainability.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .. import config
from ..explainability.gradcam import GradCAM
from .base import EmbeddingModel, EmbeddingModelInfo

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EfficientNetEmbeddingModel(EmbeddingModel):
    info = EmbeddingModelInfo(
        name="efficientnet",
        display_name="EfficientNet-B0",
        dimension=1280,
        description="Convolutional ImageNet feature extractor - Grad-CAM explainable.",
        explainability="gradcam",
    )

    _model: nn.Module | None = None

    def _load(self) -> nn.Module:
        if EfficientNetEmbeddingModel._model is None:
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            net.classifier = nn.Identity()
            EfficientNetEmbeddingModel._model = net.to(config.DEVICE).eval()
        return EfficientNetEmbeddingModel._model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)

    def embed(self, image: Image.Image) -> np.ndarray:
        model = self._load()
        tensor = self._preprocess(image)
        with torch.no_grad():
            features = model(tensor)
        vector = features[0].cpu().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def explain(self, image: Image.Image) -> np.ndarray:
        model = self._load()
        tensor = self._preprocess(image)
        target_layer = model.features[-1]
        cam = GradCAM(model, target_layer)
        try:
            return cam(tensor, model)
        finally:
            cam.remove()
