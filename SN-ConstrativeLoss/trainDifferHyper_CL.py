from trainModelCL import train_CL
from itertools import product
import numpy as np
from Plotting import loss_acc_plt

possible_activation_func = ["relu", "selu"]
margin_values = np.round(np.arange(0.1, 1.1, 0.1), 1).tolist()

possible_combos = list(product(margin_values, possible_activation_func))

all_train_losses = []
all_valid_losses = []
all_train_acc = []
all_valid_acc = []
all_model_names = []

for margin, activation in possible_combos:
    train_loss, valid_loss, train_acc, valid_acc, model_name = train_CL(margin=margin, activation=activation)
    all_train_losses.append(train_loss)
    all_valid_losses.append(valid_loss)
    all_train_acc.append(train_acc)
    all_valid_acc.append(valid_acc)
    all_model_names.append(model_name)

loss_acc_plt(train_losses=all_train_losses, valid_losses=all_valid_losses,
             train_accs=all_train_acc, valid_accs=all_valid_acc, model_names=all_model_names, plot_name="SNN-CL")