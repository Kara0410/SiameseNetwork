"""Siamese embedding backend - the modernized successor to `legacy/SN-*`.

Wraps `ai_services.siamese.network.SiameseEmbeddingNet` (ResNet18 trunk + projection
head). If `ai_services.config.SIAMESE_CHECKPOINT` points at a checkpoint produced by
`python -m ai_services.siamese.train`, it is loaded; otherwise the network runs with
an ImageNet-pretrained trunk and a randomly-initialized projection head. Grad-CAM
hooks the trunk's final residual block for explainability.
"""

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .. import config
from ..explainability.gradcam import GradCAM
from ..siamese.network import SiameseEmbeddingNet
from .base import EmbeddingModel, EmbeddingModelInfo

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SiameseEmbeddingModel(EmbeddingModel):
    info = EmbeddingModelInfo(
        name="siamese",
        display_name="Siamese ResNet18",
        dimension=256,
        description="Custom Siamese network (ResNet18 trunk + projection head), "
        "trainable with contrastive/triplet loss - Grad-CAM explainable.",
        explainability="gradcam",
    )

    _model: SiameseEmbeddingNet | None = None

    def _load(self) -> SiameseEmbeddingNet:
        if SiameseEmbeddingModel._model is None:
            net = SiameseEmbeddingNet(embedding_dim=self.info.dimension)
            checkpoint = config.SIAMESE_CHECKPOINT
            if checkpoint and os.path.exists(checkpoint):
                net.load_state_dict(torch.load(checkpoint, map_location=config.DEVICE))
            SiameseEmbeddingModel._model = net.to(config.DEVICE).eval()
        return SiameseEmbeddingModel._model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)

    def embed(self, image: Image.Image) -> np.ndarray:
        model = self._load()
        tensor = self._preprocess(image)
        with torch.no_grad():
            embedding = model.forward_once(tensor)
        vector = embedding[0].cpu().numpy()
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def explain(self, image: Image.Image) -> np.ndarray:
        model = self._load()
        tensor = self._preprocess(image)
        target_layer = model.backbone.layer4[-1]
        cam = GradCAM(model, target_layer)
        try:
            return cam(tensor, model.forward_once)
        finally:
            cam.remove()
