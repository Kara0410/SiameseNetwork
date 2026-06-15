"""Trains the vanilla Siamese Network (BCE loss, ReLU activations).

Runs a fixed number of training/validation epochs, prints per-epoch
metrics, saves the trained weights, and plots the loss/accuracy curves.
"""

import numpy as np
import torch
from tqdm import tqdm
from torch import optim

# custom created stuff
from config import DEVICE
from DatasetVanilla import train_dataloader, valid_dataloader
from SiameseNetworkVanilla import SiameseNetworkVanilla
from Plotting import loss_acc_plt

LEARNING_RATE = 1e-6
NUM_EPOCHS = 5
# Predictions above this probability are treated as "same person".
MATCH_THRESHOLD = 0.6
MODEL_NAME = "SiameseVanillaModelReLU"

# Measuring the Model
train_loss_history = []
train_acc_history = []

valid_loss_history = []
valid_acc_history = []

# tqdm stuff
# number of sample per batch
num_samples_train = len(train_dataloader)
num_samples_valid = len(valid_dataloader)

desc_1 = "Train-Iteration over the batches of Epoch "
desc_2 = "Validation-Iteration over the batches of Epoch "

# setting up the network, loss, opt
model = SiameseNetworkVanilla().to(DEVICE)
loss = torch.nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Iterate through the epochs
for epoch in range(NUM_EPOCHS):

    # Training
    model.train()
    epoch_train_loss = []
    correct_train = 0
    total_train = 0
    for anchor_img, other_img, label in tqdm(train_dataloader, total=num_samples_train, desc=desc_1 + f"{epoch}"):
        anchor_img, other_img, label = anchor_img.to(DEVICE), other_img.to(DEVICE), label.to(DEVICE)

        # Zero the gradients
        opt.zero_grad()
        # Pass in the two images into the network and obtain two outputs
        prediction = model(anchor_img, other_img)

        # Pass the outputs of the networks and label into the loss function
        loss_bce = loss(prediction, label)
        epoch_train_loss.append(loss_bce.item())

        # Accuracy calculation
        predicted_labels = (prediction > MATCH_THRESHOLD).float()
        correct_train += (predicted_labels == label).sum().item()
        total_train += label.size(0)

        # Calculate the backpropagation
        loss_bce.backward()
        # Optimize
        opt.step()

    # Calculate training accuracy
    training_accuracy = correct_train / total_train
    train_acc_history.append(training_accuracy)

    # Validation
    model.eval()
    epoch_val_loss = []
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for i, (anchor_img, other_img, label) in tqdm(enumerate(valid_dataloader, 0), total=num_samples_valid,
                                                      desc=desc_2 + f"{epoch}"):
            anchor_img, other_img, label = anchor_img.to(DEVICE), other_img.to(DEVICE), label.to(DEVICE)

            prediction = model(anchor_img, other_img)

            loss_bce = loss(prediction, label)
            epoch_val_loss.append(loss_bce.item())

            # Accuracy calculation
            predicted_labels = (prediction > MATCH_THRESHOLD).float()
            correct_val += (predicted_labels == label).sum().item()
            total_val += label.size(0)

    # Calculate validation accuracy
    validation_accuracy = correct_val / total_val
    valid_acc_history.append(validation_accuracy)

    # Print out the loss of each epoch for both training and validation
    print(f"\nEpoch number {epoch}\n"
          f"Training Loss: {np.mean(epoch_train_loss)}\n"
          f"Training Accuracy: {training_accuracy}\n"
          f"Validation Loss: {np.mean(epoch_val_loss)}\n"
          f"Validation Accuracy: {validation_accuracy}\n")

    train_loss_history.append(np.mean(epoch_train_loss))
    valid_loss_history.append(np.mean(epoch_val_loss))

# Save the trained model
torch.save(model.state_dict(), f"{MODEL_NAME}.pth")

# Plot the training and validation loss/accuracy
loss_acc_plt(train_losses=[train_loss_history], valid_losses=[valid_loss_history],
             train_accs=[train_acc_history], valid_accs=[valid_acc_history],
             model_names=[MODEL_NAME], plot_name=MODEL_NAME, epochs=NUM_EPOCHS)
