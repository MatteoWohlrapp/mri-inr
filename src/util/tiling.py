"""
Util functions for tiling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import torch


def image_to_patches(tensor, outer_patch_size, inner_patch_size):
    """
    Extract overlapping patches from a batch of grayscale images, handling variable image sizes in a batch. We work with bigger latent codes than what is predicted by the model, so we need to extract overlapping patches from the images with an outer and inner patch size.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
        outer_patch_size (int): Total size of each patch including the border.
        inner_patch_size (int): Size of the core area of each patch.

    Returns:
        Tuple[torch.Tensor, List[Tuple[int, int]]]: Tuple of extracted patches and list of tuples with number of patches (num_patches_vertical, num_patches_horizontal) for each image.
    """
    tensor = tensor.unsqueeze(1)
    batch_size, _, _, _ = tensor.shape

    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    image_information = []

    all_patches = []

    for i in range(batch_size):
        height, width = tensor[i].shape[1:3]
        vertical_pad = (
            inner_patch_size - (height % inner_patch_size)
        ) % inner_patch_size
        horizontal_pad = (
            inner_patch_size - (width % inner_patch_size)
        ) % inner_patch_size

        padded_tensor = F.pad(
            tensor[i].unsqueeze(0),
            (padding, padding + horizontal_pad, padding, padding + vertical_pad),
            mode="reflect",
        )

        patches = F.unfold(
            padded_tensor,
            kernel_size=(outer_patch_size, outer_patch_size),
            stride=stride,
        )
        num_patches_vertical = (height + vertical_pad) // inner_patch_size
        num_patches_horizontal = (width + horizontal_pad) // inner_patch_size
        image_information.append((num_patches_vertical, num_patches_horizontal))

        patches = (
            patches.transpose(1, 2)
            .contiguous()
            .view(1, -1, outer_patch_size, outer_patch_size)
        )
        all_patches.append(patches.squeeze(0))

    cat_patches = torch.cat(all_patches, dim=0)

    return torch.cat(all_patches, dim=0), image_information


def generate_weight_matrix(tile_size):
    """
    Generate a weight matrix for weighted averaging of overlapping patches.

    Args:
        tile_size (int): Size of the tile.

    Returns:
        torch.Tensor: Weight matrix for weighted averaging.
    """
    center = (tile_size - 1) / 2
    weight_matrix = torch.zeros((tile_size, tile_size))

    for i in range(tile_size):
        for j in range(tile_size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            weight = np.exp(-0.1 * distance)
            weight_matrix[i, j] = weight

    weight_matrix /= weight_matrix.max()

    return weight_matrix


def patches_to_image_weighted_average(
    tiles, image_information, outer_patch_size, inner_patch_size, device
):
    """
    Combine overlapping patches into a single image using weighted averaging.

    Args:
        tiles (torch.Tensor): Input tensor of shape (num_patches, outer_patch_size, outer_patch_size).
        image_information (List[Tuple[int, int]]): List of tuples with number of patches (num_patches_vertical, num_patches_horizontal) for each image.
        outer_patch_size (int): Total size of each patch including the border.
        inner_patch_size (int): Size of the core area of each patch.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Reconstructed image.
    """
    # Assuming a single-channel image. Adjust for multi-channel images.
    output_size = (
        image_information[0][0] * inner_patch_size,
        image_information[0][1] * inner_patch_size,
    )
    kernel_size = outer_patch_size
    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    weights = generate_weight_matrix(kernel_size).to(device)

    tiles = tiles * weights
    normalization = torch.ones_like(tiles) * weights

    flat_tiles = tiles.reshape(-1, kernel_size * kernel_size).permute(1, 0)
    flat_normalization = normalization.reshape(-1, kernel_size * kernel_size).permute(
        1, 0
    )
    image = F.fold(
        flat_tiles,
        output_size,
        kernel_size=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
    )
    normalization = F.fold(
        flat_normalization,
        output_size,
        kernel_size=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
    )
    image /= normalization

    return image


def patches_to_image(tiles, image_information, outer_patch_size, inner_patch_size):
    """
    Combine overlapping patches into a single image.

    Args:
        tiles (torch.Tensor): Input tensor of shape (num_patches, outer_patch_size, outer_patch_size).
        image_information (List[Tuple[int, int]]): List of tuples with number of patches (num_patches_vertical, num_patches_horizontal) for each image.
        outer_patch_size (int): Total size of each patch including the border.
        inner_patch_size (int): Size of the core area of each patch.

    Returns:
        torch.Tensor: Reconstructed image.
    """
    # Assuming a single-channel image. Adjust for multi-channel images.
    output_size = (
        image_information[0][0] * inner_patch_size,
        image_information[0][1] * inner_patch_size,
    )
    kernel_size = outer_patch_size
    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    flat_tiles = tiles.reshape(-1, kernel_size * kernel_size).permute(1, 0)
    image = F.fold(
        flat_tiles,
        output_size,
        kernel_size=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
    )
    normalization = F.fold(
        torch.ones_like(flat_tiles),
        output_size,
        kernel_size=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
    )
    image /= normalization

    return image


def classify_patches(tile: torch.Tensor):
    """
    Seperate into just black(mainly at the conrners) and actual data

    Args:
        tile (torch.Tensor): Input tensor of shape (1, height, width).

    Returns:
        int: 0 if the tile is black, 1 otherwise.
    """
    mean = tile.mean()
    if mean < 1e-10:
        return 0
    else:
        return 1


def filter_black_patches(
    undersampled: list[torch.tensor], fullysampled: list[torch.tensor]
):
    """
    Filter out tiles that are classified as 0 or black.

    Args:
        undersampled (list[torch.Tensor]): List of undersampled tiles.
        fullysampled (list[torch.Tensor]): List of fullysampled tiles.

    Returns:
        tuple: A tuple containing undersampled and fullysampled tiles with black tiles removed.
    """
    for i in range(len(undersampled)):
        non_black_indices = [
            index
            for index, u_tile in enumerate(undersampled[i])
            if classify_patches(u_tile) != 0
        ]
        undersampled[i] = undersampled[i][non_black_indices]
        fullysampled[i] = fullysampled[i][non_black_indices]
    return undersampled, fullysampled


def filter_and_remember_black_patches(patches):
    """
    Filters out black patches from a batch and remembers their positions.

    Args:
        patches (torch.Tensor): Batch of patches with shape (N, C, H, W).

    Returns:
        tuple: A tuple containing:
            - Non-black patches (torch.Tensor).
            - Indices of black patches (list).
            - Original shape of the batch (tuple).
    """
    non_black_indices = []
    black_indices = []

    for index, patch in enumerate(patches):
        if classify_patches(patch) == 1:  # Assuming 1 means non-black
            non_black_indices.append(index)
        else:
            black_indices.append(index)

    non_black_patches = patches[non_black_indices]
    original_shape = patches.shape

    return non_black_patches, black_indices, original_shape


def reintegrate_black_patches(processed_patches, black_indices, original_shape):
    """
    Reintegrates black patches into the batch at their original positions.

    Args:
        processed_patches (torch.Tensor): Tensor of processed non-black patches.
        black_indices (list): List of indices where black patches were located.
        original_shape (tuple): Original shape of the batch (total number of patches, channels, height, width).

    Returns:
        torch.Tensor: Tensor with black patches reintegrated.
    """
    # Ensure that the shape for the full batch accounts for the processed patches' dimensions
    full_batch = torch.zeros(
        (original_shape[0], *processed_patches.shape[1:]),
        dtype=processed_patches.dtype,
        device=processed_patches.device,
    )
    non_black_index = 0

    for i in range(original_shape[0]):
        if i in black_indices:
            # Insert a black patch with the same dimensions as the processed patches
            full_batch[i] = torch.zeros_like(processed_patches[0])
        else:
            # Insert a processed patch
            full_batch[i] = processed_patches[non_black_index]
            non_black_index += 1

    return full_batch


def extract_center_batch(batch, outer_patch_size, inner_patch_size) -> torch.Tensor:
    """
    Extract the center of a batch of tensors with the given outer patch size, to the size of the inner patch size.

    Args:
        batch (torch.Tensor): Input batch of tensors of shape (batch_size, outer_patch_size, outer_patch_size).
        outer_patch_size (int): Total size of the outer patch.
        inner_patch_size (int): Size of the inner patch.

    Returns:
        torch.Tensor: Extracted center tensor of shape (batch_size, inner_patch_size, inner_patch_size).
    """
    padding = (outer_patch_size - inner_patch_size) // 2
    center = batch[
        :, padding : padding + inner_patch_size, padding : padding + inner_patch_size
    ]
    return center


# UNUSED
def collate_fn(batch):
    """
    Collate function to combine tiles into a single tensor.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): List of tuples containing fullysampled and undersampled tiles.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing fullysampled and undersampled tiles as a single tensor.
    """
    # Each element in batch is a tuple (tiles_fullysampled, tiles_undersampled)
    # where tiles_fullysampled and tiles_undersampled are tensors of shape (num_tiles, tile_height, tile_width)

    # Separate the fullysampled and undersampled tiles
    tiles_fullysampled, tiles_undersampled = zip(*batch)

    # Flatten the list of fullysampled tiles and stack them into a single tensor
    tiles_fullysampled = torch.cat(
        [tile.view(-1, *tile.shape[1:]) for tile in tiles_fullysampled]
    )

    # Flatten the list of undersampled tiles and stack them into a single tensor
    tiles_undersampled = torch.cat(
        [tile.view(-1, *tile.shape[1:]) for tile in tiles_undersampled]
    )

    return tiles_fullysampled, tiles_undersampled
