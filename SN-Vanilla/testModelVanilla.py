import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

from DatasetVanilla import test_dataloader
from SiameseNetworkVanilla import SiameseNetworkVanilla


# Creating some helper functions
def imshow(img, text=None):
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

model = SiameseNetworkVanilla().cuda()
model.load_state_dict(torch.load("./SiameseVanillaModel.pth"))
model.eval()

for i in range(10):
    # Iterate over 10 images and test them with the first image (x0)
    _, x1, label2 = next(dataiter)

    # Concatenate the two images together
    concatenated = torch.cat((x0, x1), 0)

    prediction = model(x0.cuda(), x1.cuda())
    imshow(torchvision.utils.make_grid(concatenated), f'Similarity: {prediction.item():.2f}')
