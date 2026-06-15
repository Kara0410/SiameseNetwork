"""Dataset and dataloaders for the Contrastive-Loss Siamese Network.

Unlike the vanilla/triplet variants, this dataset draws pairs directly
from an ``ImageFolder`` over the whole data directory and labels a pair
``1`` if the two images belong to different classes (people) and ``0``
if they belong to the same class.
"""

import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from config import DATA_DIR

BATCH_SIZE = 16
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2


class SiameseNetworkDataset(Dataset):
    """Random image pairs labelled by whether they belong to the same class."""

    def __init__(self, imageFolderDataset: datasets.ImageFolder, transform: transforms.Compose = None) -> None:
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a random pair of images and a same/different-class label."""
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.imageFolderDataset.imgs)


folder_dataset = datasets.ImageFolder(root=DATA_DIR)

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])

# creating dataset
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transformation)

# getting size for the sets
total_size = len(siamese_dataset)
train_size = int(TRAIN_SPLIT * total_size)
val_size = int(VAL_SPLIT * total_size)
test_size = total_size - (train_size + val_size)

# split the dataset in train val and test set
train_data, val_data, test_data = random_split(dataset=siamese_dataset, lengths=[train_size, val_size, test_size])

# create dataloader
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)
