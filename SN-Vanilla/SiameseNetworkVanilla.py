"""CNN-based Siamese Network used by the "vanilla" (BCE loss) variant.

A single convolutional tower embeds each input image; the absolute
difference between the two embeddings is fed to a classifier that
outputs the probability that the two images show the same person.
"""

import torch
import torch.nn as nn


class SiameseNetworkVanilla(nn.Module):
    """Siamese network trained with binary cross-entropy loss."""

    def __init__(self) -> None:
        super(SiameseNetworkVanilla, self).__init__()

        self.convolutionalLayer = nn.Sequential(
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
            nn.Flatten(),

            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single image batch into the 4096-dim feature space."""
        return self.convolutionalLayer(x)

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
        """Return the probability that ``input_1`` and ``input_2`` match.

        The L1 (absolute) distance between the two embeddings is used as
        the similarity signal for the classifier.
        """
        embedded_output_1 = self.forward_once(input_1)
        embedded_output_2 = self.forward_once(input_2)

        siamese_output = self.classifier(torch.abs(embedded_output_1 - embedded_output_2))
        return siamese_output
