import matplotlib.pyplot as plt
import numpy as np

def show_image(image, cmap="gray"):
    """
    Display a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
    """
    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis("off")
    plt.show()

def save_image(image, path, cmap="gray"):
    """
    Save a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        path (str): The path to save the image to.
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
    """
    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis("off")
    plt.savefig(path)

def show_images(images, cmap="gray", ncols=2, labels=None):
    """
    Display multiple images using matplotlib.

    Args:
        images (list): A list of images as 2D tensors (height x width).
        cmap (str): The colormap to use for displaying the images. Default is 'gray'.
        ncols (int): The number of columns to use for displaying the images. Default is 2.
        labels (list): A list of labels for the images. Default is None.
    """
    nrows = (len(images) + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(5, 10))
    for i, image in enumerate(images):
        ax[i // ncols, i % ncols].imshow(image.squeeze(), cmap=cmap)
        ax[i // ncols, i % ncols].axis("off")
        if labels is not None:
            ax[i // ncols, i % ncols].set_title(labels[i])
    plt.show()

def show_image_comparison(image, cmap='gray'):
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

def save_image_comparison(image, path, cmap='gray'):
    """Save undersampled, fully sampled, reconstruction, and difference images side by side."""
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    ax[0].imshow(image[0].squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("Undersampled")
    ax[0].axis("off")
    ax[1].imshow(image[1].squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title("Fully Sampled")
    ax[1].axis("off")
    ax[2].imshow(image[2].squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")
    ax[3].imshow(1 - np.abs(image[2].squeeze() - image[1].squeeze()), cmap=cmap, vmin=0, vmax=1)
    ax[3].set_title("Difference")
    ax[3].axis("off")
    plt.savefig(path)

def show_batch(batch, cmap='gray', ncols=2):
    """Show a batch of images."""
    nrows = (len(batch) + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(5,10))
    for i, image in enumerate(batch):
        ax[i // ncols, i % ncols].imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
        ax[i // ncols, i % ncols].axis("off")
    plt.show()
    