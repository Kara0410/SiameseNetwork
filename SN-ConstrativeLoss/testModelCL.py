import torch.nn.functional as F
from DatasetForCL import test_dataloader
from SiameseNNModelCL import SiameseNetworkCL
import torchvision
import torch

import numpy as np
import matplotlib.pyplot as plt

# Creating some helper functions

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Specify the index of the image you want to visualize
specific_index = 5  # Change this to the index you want to visualize

# Get the specific image from the dataloader
dataiter = iter(test_dataloader)
for idx in range(specific_index + 1):
    x0, _, _ = next(dataiter)

model = SiameseNetworkCL().cuda()
model.load_state_dict(torch.load("./ContrastiveLossModel.pth"))
model.eval()

for i in range(10):
    _, x1, label2 = next(dataiter)

    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = model(x0.cuda(), x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
