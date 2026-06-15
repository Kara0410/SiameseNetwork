"""Modernized Siamese embedding network.

Replaces the original 4-layer custom CNN trunk (see `legacy/SN-*`) with a
configurable ResNet18 backbone plus a small projection head, trained end-to-end with
contrastive or triplet loss (see `losses.py`). The resulting embeddings plug directly
into the platform's `siamese` embedding backend, FAISS vector store, and Grad-CAM
explainability pipeline.
"""

import torch
import torch.nn as nn
from torchvision import models


class SiameseEmbeddingNet(nn.Module):
    """ResNet18 trunk + MLP projection head producing unit-normalized embeddings."""

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single batch of images, L2-normalized to the unit sphere."""
        features = self.backbone(x)
        embedding = self.projection(features)
        return nn.functional.normalize(embedding, p=2, dim=1)

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a pair of image batches for use with `ContrastiveLoss`."""
        return self.forward_once(input_a), self.forward_once(input_b)
