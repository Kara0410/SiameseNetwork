"""Loss functions for training `SiameseEmbeddingNet`.

Refactored from `legacy/SN-ConstrativeLoss/ContrastiveLoss.py` and
`legacy/SN-TripletLoss` - the legacy research found triplet loss with a SeLU-activated
trunk gave the strongest separation, which motivated keeping triplet loss as the
default objective here while preserving contrastive loss as an alternative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Pulls matching-pair embeddings together and pushes non-matching pairs apart.

    `label` is `1` for dissimilar pairs and `0` for similar pairs. Dissimilar pairs
    are only penalized while their distance is below `margin`.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        distance = F.pairwise_distance(embedding_a, embedding_b)
        loss = (1 - label) * distance.pow(2) + label * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return loss.mean()


class TripletLoss(nn.Module):
    """Thin, explicit wrapper around `nn.TripletMarginLoss` for anchor/positive/negative embeddings."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self._loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return self._loss(anchor, positive, negative)
