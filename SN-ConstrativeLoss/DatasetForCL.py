import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import random
import numpy as np
from PIL import Image


# preparing the data
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


folder_dataset = datasets.ImageFolder(root="C:/Users/Boran/Desktop/Semester 5/Practical Work in AI/data")

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])

# creating dataset
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transformation)

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
