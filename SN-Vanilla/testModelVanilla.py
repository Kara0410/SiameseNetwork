"""Visualizes vanilla model predictions on pairs from the test set.

Loads the trained weights, then for ten pairs from the test dataloader
shows the two images side by side together with the predicted
similarity score.
"""

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from config import DEVICE
from DatasetVanilla import test_dataloader
from SiameseNetworkVanilla import SiameseNetworkVanilla

MODEL_NAME = "SiameseVanillaModelReLU"


def imshow(img: torch.Tensor, text: str = None) -> None:
    """Display an image grid with an optional caption."""
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Grab one image that we are going to test
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

model = SiameseNetworkVanilla().to(DEVICE)
model.load_state_dict(torch.load(f"./{MODEL_NAME}.pth", map_location=DEVICE))
model.eval()

for i in range(10):
    # Iterate over 10 images and test them with the first image (x0)
    _, x1, label2 = next(dataiter)

    # Concatenate the two images together
    concatenated = torch.cat((x0, x1), 0)

    prediction = model(x0.to(DEVICE), x1.to(DEVICE))
    imshow(torchvision.utils.make_grid(concatenated), f'Similarity: {prediction.item():.2f}')
