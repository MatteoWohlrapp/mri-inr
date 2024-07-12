"""
Functions to calculate error metrics between two images used during evaluation.
"""

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from src.util.visualization import save_image
import numpy as np
from src.util.tiling import (
    patches_to_image_weighted_average,
    patches_to_image,
    filter_and_remember_black_tiles,
    reintegrate_black_patches,
)
import os


def calculate_data_range(original, predicted):
    """
    Calculate the data range between two images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        float: The data range between the two images.
    """
    data_min = min(np.min(original), np.min(predicted))
    data_max = max(np.max(original), np.max(predicted))
    data_range = data_max - data_min

    return data_range


def calculate_psnr(original, predicted):
    """
    Calculate the PSNR between two images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        float: The PSNR between the two images.
    """
    return psnr(
        original, predicted, data_range=calculate_data_range(original, predicted)
    )


def calculate_ssim(original, predicted):
    """
    Calculate the SSIM between two images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        float: The SSIM between the two images.
    """
    return ssim(
        original, predicted, data_range=calculate_data_range(original, predicted)
    )


def calculate_nrmse(original, predicted):
    """
    Calculate the NRMSE between two images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        float: The NRMSE between the two images.
    """
    return nrmse(original, predicted)


def calculate_difference(original, predicted):
    """
    Create and show an image visualizing the difference between the original and predicted images.

    Args:
        original (np.ndarray): The original image.
        predicted (np.ndarray): The predicted image.

    Returns:
        np.ndarray: The difference image.
    """
    # Compute the absolute difference image
    difference = np.abs(original - predicted)

    return difference


def error_metrics(
    model,
    output_dir,
    filename,
    fully_sampled,
    undersampled,
    img_information,
    device,
    outer_patch_size,
    inner_patch_size,
    siren_patch_size,
):
    """
    Calculate all the error metrics between the fully sampled and reconstructed images and save a folder with the differences.

    Args:
        model (torch.nn.Module): The model to use for reconstruction.
        output_dir (str): The output directory to save the results.
        filename (str): The filename to use for saving the results.
        fully_sampled (torch.Tensor): The fully sampled image.
        undersampled (torch.Tensor): The undersampled image.
        img_information (dict): The information about the image.
        device (torch.device): The device to use for computation.
        outer_patch_size (int): The size of the outer patch.
        inner_patch_size (int): The size of the inner patch.
        siren_patch_size (int): The size of the SIREN patch.
    """

    # Filter black tiles
    undersampled_filtered, filter_information, original_shape = (
        filter_and_remember_black_tiles(undersampled)
    )
    undersampled_filtered = undersampled_filtered.to(device)

    reconstructed_patches = model(undersampled_filtered)

    # Reintegrate black tiles
    reconstructed_patches = reintegrate_black_patches(
        reconstructed_patches, filter_information, original_shape
    )

    reconstructed_img = patches_to_image_weighted_average(
        reconstructed_patches,
        img_information,
        siren_patch_size,
        inner_patch_size,
        device,
    )
    undersampled_img = patches_to_image(
        undersampled, img_information, outer_patch_size, inner_patch_size
    )
    fully_sampled_img = patches_to_image(
        fully_sampled, img_information, outer_patch_size, inner_patch_size
    )

    reconstructed_img = reconstructed_img.cpu()
    fully_sampled_img = fully_sampled_img.cpu()
    undersampled_img = undersampled_img.cpu()

    save_image(reconstructed_img, f"{filename}_reconstructed", output_dir)
    save_image(undersampled_img, f"{filename}_undersampled", output_dir)
    save_image(fully_sampled_img, f"{filename}_fully_sampled", output_dir)
    save_image(
        calculate_difference(
            fully_sampled_img.squeeze().numpy(), reconstructed_img.squeeze().numpy()
        ),
        f"{filename}_difference",
        output_dir,
        cmap="viridis",
    )

    # Calculate the error metrics
    psnr_value = calculate_psnr(
        fully_sampled_img.squeeze().numpy(), reconstructed_img.squeeze().numpy()
    )
    ssim_value = calculate_ssim(
        fully_sampled_img.squeeze().numpy(), reconstructed_img.squeeze().numpy()
    )
    nrmse_value = calculate_nrmse(
        fully_sampled_img.squeeze().numpy(), reconstructed_img.squeeze().numpy()
    )

    # Write them to a file
    with open(os.path.join(output_dir, f"{filename}_error.txt"), "w") as f:
        f.write(f"PSNR: {psnr_value}\n")
        f.write(f"SSIM: {ssim_value}\n")
        f.write(f"NRMSE: {nrmse_value}\n")
