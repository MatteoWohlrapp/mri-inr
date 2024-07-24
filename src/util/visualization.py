"""
Util functions for visualizing images.
"""

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


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


def show_batch(batch, cmap="gray", ncols=2):
    """
    Show a batch of images.

    Args:
        batch (torch.Tensor): A batch of images as a 4D tensor (batch_size x channels x height x width).
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
        ncols (int): The number of columns to use for displaying the images. Default
    """
    nrows = (len(batch) + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(5, 10))
    for i, image in enumerate(batch):
        ax[i // ncols, i % ncols].imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
        ax[i // ncols, i % ncols].axis("off")
    plt.show()


def save_image_comparison(
    fully_sampled, undersampled, reconstructed, path, cmap="gray"
):
    """
    Save undersampled, fully sampled, reconstruction, and difference images side by side.

    Args:
        fully_sampled (torch.Tensor): The fully sampled image.
        undersampled (torch.Tensor): The undersampled image.
        reconstructed (torch.Tensor): The reconstructed image.
        path (str): The path to save the image.
        cmap (str): The colormap to use for displaying the image. Default is
    """
    plt.clf()
    fully_sampled = fully_sampled.cpu()
    undersampled = undersampled.cpu()
    reconstructed = reconstructed.cpu()

    fully_sampled = normalize_scan(fully_sampled)
    undersampled = normalize_scan(undersampled)
    reconstructed = normalize_scan(reconstructed)

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    ax[0].imshow(undersampled.squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("Undersampled")
    ax[0].axis("off")
    ax[1].imshow(fully_sampled.squeeze(), cmap=cmap, vmin=0, vmax=1)
    ax[1].set_title("Fully Sampled")
    ax[1].axis("off")
    ax[2].imshow(reconstructed.squeeze(), cmap=cmap)
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")
    ax[3].imshow(
        1 - np.abs(reconstructed.squeeze() - fully_sampled.squeeze()),
        cmap="viridis",
    )
    ax[3].set_title("Difference")
    ax[3].axis("off")
    plt.savefig(path)


def save_image(image, filename, output_dir, cmap="gray"):
    """
    Save a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        filename (str): The filename to save the image as.
        output_dir (str): The directory to save the image in.
    """
    plt.clf()
    image = normalize_scan(image)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    im = plt.imshow(image.squeeze(), cmap=cmap, vmin=0, vmax=1)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(
        f"{output_dir}/{filename}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1200,
    )
    plt.close()


def normalize_scan(scan: torch.Tensor) -> torch.Tensor:
    """
    Normalize the MRI scan.

    Args:
        scan (torch.Tensor): The MRI scan to normalize.

    Returns:
        torch.Tensor: The normalized MRI scan.
    """
    scan_min = scan.min()
    scan_max = scan.max()
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan


def metrics_boxplot(metrics, output_dir, suffix=None):
    """
    Create a boxplot of the metrics.

    Args:
        metrics (dict): The metrics to plot.
        output_dir (str): The directory to save the plot in.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, value in metrics.items():
        fig, ax = plt.subplots()
        ax.boxplot(value)
        ax.set_xticklabels([key])
        ax.set_title(key)
        plt.savefig(f"{output_dir}/{key}_metrics_boxplot.png")
        plt.close()


def metrics_density_plot(metrics, output_dir, suffix=None):
    """
    Create a density plot of the metrics.

    Args:
        metrics (dict): The metrics to plot.
        output_dir (str): The directory to save the plot in.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, value in metrics.items():
        fig, ax = plt.subplots()
        ax = sns.kdeplot(value)
        ax.set_title(key)
        plt.savefig(f"{output_dir}/{key}_density_plot.png")
        plt.close()
