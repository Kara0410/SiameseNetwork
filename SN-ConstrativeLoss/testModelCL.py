"""Visualizes Contrastive-Loss model predictions on pairs from the test set.

Loads a trained checkpoint (named after its margin/activation
configuration, see ``trainModelCL.train_CL``), then for ten pairs from
the test dataloader shows the two images together with their embedding
distance ("dissimilarity").
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE
from DatasetForCL import test_dataloader
from SiameseNNModelCL import SiameseNetworkCL

# Must match the margin/activation of the checkpoint produced by train_CL,
# e.g. "CL-1-SELU" for margin=1, activation="selu".
MODEL_NAME = "CL-1-SELU"
ACTIVATION = "selu"

# Index of the first image (x0) to compare the other test pairs against.
SPECIFIC_INDEX = 5


def imshow(img: torch.Tensor, text: str = None) -> None:
    """Display an image grid with an optional caption."""
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get the specific image from the dataloader
dataiter = iter(test_dataloader)
for idx in range(SPECIFIC_INDEX + 1):
    x0, _, _ = next(dataiter)

model = SiameseNetworkCL(activation=ACTIVATION).to(DEVICE)
model.load_state_dict(torch.load(f"./{MODEL_NAME}.pth", map_location=DEVICE))
model.eval()

for i in range(10):
    _, x1, label2 = next(dataiter)

    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = model(x0.to(DEVICE), x1.to(DEVICE))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
