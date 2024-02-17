import numpy as np
import torch
from tqdm import tqdm

from torch import optim
# custom created stuff
from DatasetForCL import train_dataloader, valid_dataloader
from SiameseNNModelCL import SiameseNetworkCL
from ContrastiveLoss import ContrastiveLoss


def train_CL(margin=1, activation="selu"):
    # tqdm stuff
    # number of sample per batch
    num_samples_train = len(train_dataloader)
    num_samples_valid = len(valid_dataloader)

    desc_1 = "Train-Iteration over the batches of Epoch "
    desc_2 = "Validation-Iteration over the batches of Epoch "

    # setting up the network, loss, opt
    device = torch.device("cuda")
    model = SiameseNetworkCL(activation=activation).to(device=device)
    loss = ContrastiveLoss(margin=margin)
    opt = optim.Adam(model.parameters(), lr=1e-5)
    Epochs = 5

    # Measuring the Model
    counter = []
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    iteration_number = 0

    # Iterate through the epochs
    for epoch in range(1, Epochs + 1):

        # Training
        model.train()
        epoch_train_loss = []
        correct_train = 0
        total_train = 0
        for img0, img1, label in tqdm(train_dataloader, total=num_samples_train,
                                      desc=desc_1 + f"{epoch}"):
            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            # Zero the gradients
            opt.zero_grad()
            # Pass in the two images into the network and obtain two outputs
            output1, output2 = model(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = loss(output1, output2, label)
            epoch_train_loss.append(loss_contrastive.item())

            # Calculate accuracy
            euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)
            predictions = (euclidean_distance > 0.5).float()  # Adjust the threshold as needed
            correct_train += torch.sum(predictions == label).item()
            total_train += label.size(0)

            # Calculate the backpropagation
            loss_contrastive.backward()
            # Optimize
            opt.step()

        # Validation
        model.eval()
        epoch_val_loss = []
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for img0, img1, label in tqdm(valid_dataloader, total=num_samples_valid,
                                          desc=desc_2 + f"{epoch}"):
                # Send the images and labels to CUDA
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                output1, output2 = model(img0, img1)

                loss_contrastive = loss(output1, output2, label)
                epoch_val_loss.append(loss_contrastive.item())

                # Calculate accuracy
                euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, keepdim=True)
                predictions = (euclidean_distance > 0.5).float()  # Adjust the threshold as needed
                correct_valid += torch.sum(predictions == label).item()
                total_valid += label.size(0)

        iteration_number += 1
        counter.append(iteration_number)

        # calculating acc
        train_accuracy = correct_train / total_train
        train_acc_history.append(train_accuracy)
        valid_accuracy = correct_valid / total_valid
        valid_acc_history.append(valid_accuracy)

        # calculating loss
        train_loss_history.append(np.mean(epoch_train_loss))
        valid_loss_history.append(np.mean(epoch_val_loss))

        # Print out the loss of each epoch for both training and validation
        print(f"\nEpoch number {epoch}\n"
              f"Training Loss: {np.mean(epoch_train_loss)}\n"
              f"Training Accuracy: {train_accuracy}\n"
              f"Validation Loss: {np.mean(epoch_val_loss)}\n"
              f"Validation Accuracy: {valid_accuracy}\n")

    # Saving
    model_name = f"CL-{margin}-{(activation).upper()}"
    # Save the trained model
    #torch.save(model.state_dict(), f"{model_name}.pth")
    return train_loss_history, valid_loss_history, train_acc_history, valid_acc_history, model_name