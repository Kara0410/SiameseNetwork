import numpy as np
import torch
from tqdm import tqdm
from torch import optim

# custom created stuff
from DatasetVanilla import train_dataloader, valid_dataloader
from SiameseNetworkVanilla import SiameseNetworkVanilla
from Plotting import loss_acc_plt

# Measuring the Model
counter = []
train_loss_history = []
train_acc_history = []

valid_loss_history = []
valid_acc_history = []

iteration_number = 0

# tqdm stuff
# number of sample per batch
num_samples_train = len(train_dataloader)
num_samples_valid = len(valid_dataloader)

desc_1 = "Train-Iteration over the batches of Epoch "
desc_2 = "Validation-Iteration over the batches of Epoch "

# setting up the network, loss, opt
model = SiameseNetworkVanilla().cuda()
loss = torch.nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=1e-6)
EPOCH = 5

# Iterate through the epochs
for epoch in range(EPOCH):

    # Training
    model.train()
    epoch_train_loss = []
    correct_train = 0
    total_train = 0
    for anchor_img, other_img, label in tqdm(train_dataloader, total=num_samples_train, desc=desc_1 + f"{epoch}"):
        # Send the images and labels to CUDA
        anchor_img, other_img, label = anchor_img.cuda(), other_img.cuda(), label.cuda()

        # Zero the gradients
        opt.zero_grad()
        # Pass in the two images into the network and obtain two outputs
        prediction = model(anchor_img, other_img)

        # Pass the outputs of the networks and label into the loss function
        loss_bce = loss(prediction, label)
        epoch_train_loss.append(loss_bce.item())

        # Accuracy calculation
        predicted_labels = (prediction > 0.6).float()
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
            # Send the images and labels to CUDA
            anchor_img, other_img, label = anchor_img.cuda(), other_img.cuda(), label.cuda()

            prediction = model(anchor_img, other_img)

            loss_bce = loss(prediction, label)
            epoch_val_loss.append(loss_bce.item())

            # Accuracy calculation
            predicted_labels = (prediction > 0.6).float()
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

    iteration_number += 1
    counter.append(iteration_number)
    train_loss_history.append(np.mean(epoch_train_loss))
    valid_loss_history.append(np.mean(epoch_val_loss))

# Save the trained model
torch.save(model.state_dict(), "SiameseVanillaModelReLU.pth")


# Plot the training and validation loss
loss_acc_plt(train_losses=train_loss_history, valid_losses=valid_loss_history,
             train_accs=train_acc_history, valid_accs=valid_acc_history, model_names="SV-ReLU")
