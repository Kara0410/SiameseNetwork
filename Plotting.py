import matplotlib.pyplot as plt

def loss_acc_plt(train_losses: list, valid_losses: list,
                 train_accs: list, valid_accs: list,
                 model_names, epochs=5,):
    # Calculate the number of rows needed for subplots
    num_plots = len(train_losses)
    num_cols = 4  # Change the number of columns to 4
    num_rows = num_plots // num_cols
    if num_plots % num_cols != 0:
        num_rows += 1

    # Define the range for epochs
    epochs_range = list(range(1, epochs + 1))

    # Create two figures, one for loss and one for accuracy
    fig_loss, axs_loss = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2.5), sharex=True, sharey=True)
    fig_acc, axs_acc = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2.5), sharex=True, sharey=True)

    # Flatten the axes array for easy iteration
    axs_loss = axs_loss.ravel()
    axs_acc = axs_acc.ravel()

    # Iterate over the number of models to create loss plots
    for i in range(num_plots):
        loss_ax = axs_loss[i]
        loss_ax.plot(epochs_range, train_losses[i], 'g-', label='Training')
        loss_ax.plot(epochs_range, valid_losses[i], 'r-', label='Validation')
        loss_ax.set_title(f'{model_names[i]} Loss')
        loss_ax.set_ylabel('Loss')
        loss_ax.legend(loc='upper right')

    # Iterate over the number of models to create accuracy plots
    for i in range(num_plots):
        acc_ax = axs_acc[i]
        acc_ax.plot(epochs_range, train_accs[i], 'b--', label='Training')
        acc_ax.plot(epochs_range, valid_accs[i], 'orange', label='Validation')
        acc_ax.set_title(f'{model_names[i]} Accuracy')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.legend(loc='lower right')

    # Set a common X label for both figures
    fig_loss.text(0.5, 0.04, 'Epochs', ha='center', va='center')
    fig_acc.text(0.5, 0.04, 'Epochs', ha='center', va='center')

    # Adjust the layout for both figures
    fig_loss.tight_layout()
    fig_acc.tight_layout()

    # Save the plots as separate images
    fig_loss.savefig(f"{model_names}_loss.png")
    fig_acc.savefig(f"{model_names}_acc.png")

"""
# Example usage with dummy data
# You should replace the following lists with your actual data.
train_losses = [[0.4, 0.35, 0.3, 0.25, 0.2]] * 20
valid_losses = [[0.45, 0.4, 0.35, 0.3, 0.25]] * 20
train_accs = [[0.6, 0.65, 0.7, 0.75, 0.8]] * 20
valid_accs = [[0.55, 0.6, 0.65, 0.7, 0.75]] * 20
model_names = [f"Model_{i+1}" for i in range(20)]

loss_acc_plt(train_losses, valid_losses, train_accs, valid_accs, model_names, epochs=5, plot_name="model_performance")
"""