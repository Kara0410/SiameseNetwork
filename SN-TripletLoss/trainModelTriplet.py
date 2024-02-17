import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


from torch import optim
# custom created stuff
from DatasetTriplet import train_dataloader, valid_dataloader
from SiameseNNModelTriplet import SiameseNetworkTriplet

# tqdm stuff
# number of sample per batch
num_samples_train = len(train_dataloader)
num_samples_valid = len(valid_dataloader)

desc_1 = "Train-Iteration over the batches of Epoch "
desc_2 = "Validation-Iteration over the batches of Epoch "


def train_Triplet(margin=1, activation_func="selu"):
    # setting up the network, loss, opt
    model = SiameseNetworkTriplet(activation_func=activation_func).cuda()
    loss = torch.nn.TripletMarginLoss(margin=margin)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    EPOCH = 5

    # Measuring the Model
    counter = []
    train_loss_history = []
    valid_loss_history = []

    train_acc_history = []
    valid_acc_history = []
    iteration_number = 0

    # Iterate through the epochs
    for epoch in range(EPOCH):

        # Training
        model.train()
        epoch_train_loss = []
        train_correct = 0
        train_total = 0
        for i, (anc_img, pos_img, neg_img) in tqdm(enumerate(train_dataloader, 0), total=num_samples_train,
                                                   desc=desc_1 + f"{epoch}"):
            # Send the images and labels to CUDA
            anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()

            # Zero the gradients
            opt.zero_grad()
            # Pass in the two images into the network and obtain two outputs
            output1, output2, output3 = model(anc_img, pos_img, neg_img)

            # Pass the outputs of the networks and label into the loss function
            triplet_loss = loss(output1, output2, output3)
            epoch_train_loss.append(triplet_loss.item())

            # Calculate the Euclidean distances between embeddings
            pos_dist = F.pairwise_distance(output1, output2)
            neg_dist = F.pairwise_distance(output1, output3)

            # Accuracy is the number of correct predictions (lower distance for positive pairs, higher distance for negative pairs)
            train_correct += torch.sum(pos_dist < neg_dist).item()
            train_total += pos_dist.size(0)

            # Calculate the backpropagation
            triplet_loss.backward()
            # Optimize
            opt.step()

        train_acc = train_correct / train_total
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        epoch_val_loss = []
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for i, (anc_img, pos_img, neg_img) in tqdm(enumerate(valid_dataloader, 0), total=num_samples_valid,
                                                       desc=desc_2 + f"{epoch}"):
                # Send the images and labels to CUDA
                anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()

                output1, output2, output3 = model(anc_img, pos_img, neg_img)

                triplet_loss = loss(output1, output2, output3)
                epoch_val_loss.append(triplet_loss.item())

                # Calculate the Euclidean distances between embeddings
                pos_distance = F.pairwise_distance(output1, output2)
                neg_distance = F.pairwise_distance(output1, output3)

                # Accuracy is the number of correct predictions (lower distance for positive pairs, higher distance for negative pairs)
                valid_correct += torch.sum(pos_distance < neg_distance).item()
                valid_total += pos_distance.size(0)

            valid_acc = valid_correct / valid_total
            valid_acc_history.append(valid_acc)


        # Print out the loss of each epoch for both training and validation
        print(f"\nEpoch number {epoch}\n"
              f"Training Loss: {np.mean(epoch_train_loss)}\n"
              f"Training Accuracy: {train_acc}\n"
              f"Validation Loss: {np.mean(epoch_val_loss)}\n"
              f"Validation Accuracy: {valid_acc}\n")

        iteration_number += 1
        counter.append(iteration_number)
        train_loss_history.append(np.mean(epoch_train_loss))
        valid_loss_history.append(np.mean(epoch_val_loss))

    # Saving
    model_name = f"TL-{margin}-{activation_func}"

    # Save the trained model
    # torch.save(model.state_dict(), f"{model_name}.pth")
    return train_loss_history, valid_loss_history, train_acc_history, valid_acc_history, model_name

