import torch
import torch.nn.functional as F
import numpy as np
import torch


def extract_with_inner_patches(tensor, outer_patch_size, inner_patch_size):
    """
    Extract overlapping patches from a batch of grayscale images, handling variable image sizes in a batch.

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

    return torch.cat(all_patches, dim=0)


def extract_with_inner_patches_with_info(tensor, outer_patch_size, inner_patch_size):
    """
    Extract overlapping patches from a batch of grayscale images, handling variable image sizes in a batch.

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


def classify_tile(tile: torch.Tensor):
    """Seperate into just black(mainly at the conrners) and actual data"""
    mean = tile.mean()
    if mean < 1e-10:
        return 0
    else:
        return 1


def filter_black_tiles(undersampled, fullysampled):
    """Filter out tiles that are classified as 0 or black."""
    non_black_tiles = [
        (u_tile, f_tile)
        for u_tile, f_tile in zip(undersampled, fullysampled)
        if classify_tile(u_tile) != 0
    ]
    return (
        torch.stack([tile[0] for tile in non_black_tiles]),
        torch.stack([tile[1] for tile in non_black_tiles]),
    )


def collate_fn(batch):
    """Collate function to combine tiles into a single tensor."""
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


def extract_center(tensor, outer_patch_size, inner_patch_size) -> torch.Tensor:
    """
    Extract the center of a tensor with the given outer patch size, to the size of the inner patch size.

    Args:
        tensor (torch.Tensor): Input tensor of shape (outer_patch_size, outer_patch_size).
        outer_patch_size (int): Total size of the outer patch.
        inner_patch_size (int): Size of the inner patch.

    Returns:
        torch.Tensor: Extracted center tensor of shape (inner_patch_size, inner_patch_size).
    """
    padding = (outer_patch_size - inner_patch_size) // 2
    center = tensor[
        padding : padding + inner_patch_size, padding : padding + inner_patch_size
    ]
    return center


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


def alternative_image_to_tiles(image: torch.Tensor, outer_patch_size, inner_patch_size):
    kernel_size = outer_patch_size
    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    patches = F.unfold(
        image, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding
    )
    patches = (
        patches.transpose(0, 1)
        .contiguous()
        .view(-1, 1, outer_patch_size, outer_patch_size)
    )
    info = (image.shape[1] // inner_patch_size, image.shape[2] // inner_patch_size)
    return patches, info


def alternative_tiles_to_image(
    tiles, image_information, outer_patch_size, inner_patch_size
):
    # Assuming a single-channel image. Adjust for multi-channel images.
    output_size = (
        image_information[0] * inner_patch_size,
        image_information[1] * inner_patch_size,
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


def alternative_tiles_to_image2(
    tiles, image_information, outer_patch_size, inner_patch_size
):
    # Assuming a single-channel image. Adjust for multi-channel images.
    output_size = (
        image_information[0] * inner_patch_size,
        image_information[1] * inner_patch_size,
    )
    kernel_size = outer_patch_size
    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    weights = generate_weight_matrix(kernel_size)

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


def generate_weight_matrix(tile_size):
    center = (tile_size - 1) / 2
    weight_matrix = torch.zeros((tile_size, tile_size))

    for i in range(tile_size):
        for j in range(tile_size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            weight = np.exp(-0.1 * distance)
            weight_matrix[i, j] = weight

    weight_matrix /= weight_matrix.max()

    return weight_matrix
