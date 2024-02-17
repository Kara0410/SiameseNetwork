import os
import torch
import random
import numpy as np
from PIL import Image
from itertools import product
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

from CreateImgDirectories import pos_path, neg_path, anc_path


# preparing the data
class SiameseNetworkDataset(Dataset):
    def __init__(self, pos_path, anc_path, neg_path):
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

    def __getitem__(self, index):
        anc_img, other_img, label = self.dataset[index]

        # getting the Pil Image
        anc_img = Image.open(anc_img).convert("RGB")
        other_img = Image.open(other_img).convert("RGB")

        # transforming the Pil to tensor and return it plus with the label
        return self.transform(anc_img), self.transform(other_img), torch.from_numpy(np.array([label], dtype=np.float32))

    def __len__(self):
        return len(self.dataset)


# creating dataset
siamese_dataset = SiameseNetworkDataset(pos_path=pos_path, neg_path=neg_path, anc_path=anc_path)

# getting size for the sets
total_size = len(siamese_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - (train_size + val_size)

# split the dataset in train val and test set
train_data, val_data, test_data = random_split(dataset=siamese_dataset, lengths=[train_size, val_size, test_size])

# create dataloader
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
valid_dataloader = DataLoader(val_data, shuffle=False, batch_size=16)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)
