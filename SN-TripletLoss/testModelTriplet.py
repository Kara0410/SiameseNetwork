import torch.nn.functional as F
from DatasetTriplet import test_dataloader
from SiameseNNModelTriplet import SiameseNetworkTriplet
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


# Specify the index of the triplet you want to visualize
specific_index = 5  # Change this to the index you want to visualize

# Get the specific triplet from the dataloader
dataiter = iter(test_dataloader)
for idx in range(specific_index + 1):
    x0, x1, _ = next(dataiter)

model = SiameseNetworkTriplet().cuda()
model.load_state_dict(torch.load("./TripletLossModel.pth"))
model.eval()

for i in range(10):
    _, x1, x2 = next(dataiter)

    output1, output2, output3 = model(x0.cuda(), x1.cuda(), x2.cuda())

    # You can choose different dissimilarity measures based on your loss function
    dissimilarity_pos = F.pairwise_distance(output1, output2)
    dissimilarity_neg = F.pairwise_distance(output1, output3)

    # Visualize the triplets with dissimilarity information
    concatenated = torch.cat((x0, x1, x2), 0)
    imshow(torchvision.utils.make_grid(concatenated),
           f'Dissimilarity (Pos): {dissimilarity_pos.item():.2f}, Dissimilarity (Neg): {dissimilarity_neg.item():.2f}')
