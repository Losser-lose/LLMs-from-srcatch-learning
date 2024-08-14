import matplotlib.pyplot as plt
import os


def plot_values(epochs_seen, examples_seen, train_values, val_values, label, save_dir=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    if save_dir:
        save_path = os.path.join(save_dir, f"{label}-plot.png")
        plt.savefig(save_path)
    plt.show()
    