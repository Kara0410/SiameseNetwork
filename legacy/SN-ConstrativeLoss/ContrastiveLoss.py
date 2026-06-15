"""Contrastive loss for training the SN-ConstrativeLoss Siamese Network.

References:
    https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
    https://lilianweng.github.io/posts/2021-05-31-contrastive/
"""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Pulls embeddings of matching pairs together and pushes non-matching
    pairs apart by at least ``margin``."""

    def __init__(self, margin: float = 1) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out_1: torch.Tensor, out_2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute the contrastive loss between two embedding batches.

        ``label`` is 1 when the pair is dissimilar and 0 when it is
        similar. Dissimilar pairs are only penalized while their
        distance is below ``margin`` (the clamp), so the loss stops
        pushing them apart once they're sufficiently separated.
        """
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2, keepdim=True)

        contrastiveLoss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) *
            torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return contrastiveLoss
