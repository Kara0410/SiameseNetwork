"""Shared plotting helper for visualizing training/validation curves.

Used by the training scripts of all three Siamese Network variants to
compare loss and accuracy across one or more model configurations.
"""

import matplotlib.pyplot as plt


def loss_acc_plt(
    train_losses: list,
    valid_losses: list,
    train_accs: list,
    valid_accs: list,
    model_names: list,
    plot_name: str = "model",
    epochs: int = 5,
) -> None:
    """Plot per-epoch loss and accuracy curves for one or more models.

    Each of ``train_losses``, ``valid_losses``, ``train_accs`` and
    ``valid_accs`` is a list with one entry per model, where each entry
    is itself a list of per-epoch values. A grid of subplots (one per
    model) is created for the loss curves and another for the accuracy
    curves, then both figures are saved to disk.

    Args:
        train_losses: Per-model lists of training loss per epoch.
        valid_losses: Per-model lists of validation loss per epoch.
        train_accs: Per-model lists of training accuracy per epoch.
        valid_accs: Per-model lists of validation accuracy per epoch.
        model_names: Display name for each model, used as subplot titles.
        plot_name: Base filename used for the saved figures
            (``f"{plot_name}_loss.png"`` and ``f"{plot_name}_acc.png"``).
        epochs: Number of epochs each model was trained for.
    """
    # Lay out the per-model subplots in a grid with up to 4 columns.
    num_plots = len(train_losses)
    num_cols = 4
    num_rows = num_plots // num_cols
    if num_plots % num_cols != 0:
        num_rows += 1

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
    fig_loss.savefig(f"{plot_name}_loss.png")
    fig_acc.savefig(f"{plot_name}_acc.png")
