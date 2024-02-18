"""CNN-based Siamese Network used by the Contrastive-Loss variant.

Supports swapping the convolutional trunk's activation function between
ReLU and SeLU so the two configurations can be compared under an
otherwise identical architecture.
"""

import torch
import torch.nn as nn


class SiameseNetworkCL(nn.Module):
    """Siamese network whose embeddings are compared via Contrastive Loss."""

    def __init__(self, activation: str = "relu") -> None:
        super(SiameseNetworkCL, self).__init__()

        if activation not in ("relu", "selu"):
            raise ValueError(f"Unsupported activation '{activation}', expected 'relu' or 'selu'")

        self.activation = activation

        self.convolutionalLayerRelu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(10, 10)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.ReLU(),
            nn.Flatten()
        )

        self.convolutionalLayerSelu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(10, 10)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=(7, 7)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=(4, 4)),
            nn.SELU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.SELU(),
            nn.Flatten()
        )

        self.connectedLayer = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single image batch using the configured activation trunk."""
        if self.activation == "relu":
            x = self.convolutionalLayerRelu(x)
        else:
            x = self.convolutionalLayerSelu(x)
        x = self.connectedLayer(x)
        return x

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed both inputs; the embeddings are compared by ContrastiveLoss."""
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        return output_1, output_2
