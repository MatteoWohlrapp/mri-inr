import matplotlib.pyplot as plt
import numpy as np
import os


def show_image(image, cmap="gray"):
    """
    Display a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
    """
    plt.imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    plt.show()


def show_image_comparison(image, cmap="gray"):
    """Show undersampled and fully sampled images side by side."""
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    ax[0].imshow(image[0].squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("Undersampled")
    ax[0].axis("off")
    ax[1].imshow(image[1].squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title("Fully Sampled")
    ax[1].axis("off")
    if len(image) > 2:
        ax[2].imshow(image[2].squeeze(), cmap=cmap, vmin=0, vmax=1)
        ax[2].set_title("Reconstruction")
        ax[2].axis("off")
        ax[3].imshow(1 - np.abs(image[2].squeeze() - image[1].squeeze()), cmap=cmap)
        ax[3].set_title("Difference")
        ax[3].axis("off")
    plt.show()


def save_image_comparison(
    fully_sampled, undersampled, reconstructed, path, cmap="gray"
):
    """Save undersampled, fully sampled, reconstruction, and difference images side by side."""
    fully_sampled = fully_sampled.cpu()
    undersampled = undersampled.cpu()
    reconstructed = reconstructed.cpu()
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    ax[0].imshow(undersampled.squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("Undersampled")
    ax[0].axis("off")
    ax[1].imshow(fully_sampled.squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title("Fully Sampled")
    ax[1].axis("off")
    ax[2].imshow(reconstructed.squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")
    ax[3].imshow(
        1 - np.abs(reconstructed.squeeze() - fully_sampled.squeeze()),
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax[3].set_title("Difference")
    ax[3].axis("off")
    plt.savefig(path)


def show_batch(batch, cmap="gray", ncols=2):
    """Show a batch of images."""
    nrows = (len(batch) + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(5, 10))
    for i, image in enumerate(batch):
        ax[i // ncols, i % ncols].imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
        ax[i // ncols, i % ncols].axis("off")
    plt.show()

def save_image(image, filename, output_dir, cmap="gray"):
    """
    Save a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        filename (str): The filename to save the image as.
        output_dir (str): The directory to save the image in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis("off")
    plt.savefig(
        f"{output_dir}/{filename}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1200,
    )
    plt.close()

    