"""Dataset and dataloaders for the vanilla (BCE loss) Siamese Network.

Builds anchor/positive (label 1) and anchor/negative (label 0) image
pairs from the anchor/positive/negative folders and splits them into
train/validation/test sets.
"""

import os
import random
from itertools import product

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from config import ANCHOR_DIR, NEGATIVE_DIR, POSITIVE_DIR

BATCH_SIZE = 16
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2


class SiameseNetworkDataset(Dataset):
    """Pairs of (anchor, other) images with a same/different-person label."""

    def __init__(self, pos_path: str, anc_path: str, neg_path: str) -> None:
        self.pos_path, self.anc_path, self.neg_path = pos_path, anc_path, neg_path

        anc_list = [os.path.join(f"{anc_path}/{file_name}") for file_name in os.listdir(anc_path)]
        pos_list = [os.path.join(f"{pos_path}/{file_name}") for file_name in os.listdir(pos_path)]
        neg_list = [os.path.join(f"{neg_path}/{file_name}") for file_name in os.listdir(neg_path)]

        # Create a list of pairs for negative example and positive
        pos_pairs = random.choices(list(product(anc_list, pos_list, [1])), k=len(pos_list))
        neg_pairs = random.choices(list(product(anc_list, neg_list, [0])), k=len(neg_list))

        # dataset of pairs
        self.dataset = pos_pairs + neg_pairs

        self.transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (anchor, other) image tensors and their pair label."""
        anc_img, other_img, label = self.dataset[index]

        # getting the Pil Image
        anc_img = Image.open(anc_img).convert("RGB")
        other_img = Image.open(other_img).convert("RGB")

        # transforming the Pil to tensor and return it plus with the label
        return self.transform(anc_img), self.transform(other_img), torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.dataset)


# creating dataset
siamese_dataset = SiameseNetworkDataset(pos_path=POSITIVE_DIR, neg_path=NEGATIVE_DIR, anc_path=ANCHOR_DIR)

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
