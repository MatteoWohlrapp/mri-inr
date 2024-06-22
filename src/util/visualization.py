import matplotlib.pyplot as plt


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

def show_image_comparison(image, cmap='gray'):
    """Show undersampled and fully sampled images side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[0].squeeze(), cmap=cmap)
    ax[0].set_title("Undersampled")
    ax[0].axis("off")
    ax[1].imshow(image[1].squeeze(), cmap=cmap)
    ax[1].set_title("Fully Sampled")
    ax[1].axis("off")
    plt.show()

def show_batch(batch, cmap='gray', ncols=2):
    """Show a batch of images."""
    nrows = (len(batch) + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(5,10))
    for i, image in enumerate(batch):
        ax[i // ncols, i % ncols].imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
        ax[i // ncols, i % ncols].axis("off")
    plt.show()
    