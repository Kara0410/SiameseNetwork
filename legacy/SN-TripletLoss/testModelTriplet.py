"""Visualizes Triplet-Loss model predictions on triplets from the test set.

Loads a trained checkpoint (named after its margin/activation
configuration, see ``trainModelTriplet.train_Triplet``), then for ten
triplets from the test dataloader shows the anchor/positive/negative
images together with their embedding distances.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from config import DEVICE
from DatasetTriplet import test_dataloader
from SiameseNNModelTriplet import SiameseNetworkTriplet

# Must match the margin/activation of the checkpoint produced by
# train_Triplet, e.g. "TL-1-selu" for margin=1, activation_func="selu".
MODEL_NAME = "TL-1-selu"
ACTIVATION = "selu"

# Index of the first triplet (x0) to compare the other test triplets against.
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


# Get the specific triplet from the dataloader
dataiter = iter(test_dataloader)
for idx in range(SPECIFIC_INDEX + 1):
    x0, x1, _ = next(dataiter)

model = SiameseNetworkTriplet(activation_func=ACTIVATION).to(DEVICE)
model.load_state_dict(torch.load(f"./{MODEL_NAME}.pth", map_location=DEVICE))
model.eval()

for i in range(10):
    _, x1, x2 = next(dataiter)

    output1, output2, output3 = model(x0.to(DEVICE), x1.to(DEVICE), x2.to(DEVICE))

    # You can choose different dissimilarity measures based on your loss function
    dissimilarity_pos = F.pairwise_distance(output1, output2)
    dissimilarity_neg = F.pairwise_distance(output1, output3)

    # Visualize the triplets with dissimilarity information
    concatenated = torch.cat((x0, x1, x2), 0)
    imshow(torchvision.utils.make_grid(concatenated),
           f'Dissimilarity (Pos): {dissimilarity_pos.item():.2f}, Dissimilarity (Neg): {dissimilarity_neg.item():.2f}')
