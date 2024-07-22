"""
Utils function for restarting a training job from the latest checkpoint.
"""

import os
import re


def find_latest_folder(base_dir, base_name):
    """
    Find the latest directory with the base name in the base directory.

    Args:
        base_dir (str): The base directory to search in.
        base_name (str): The base name of the directory to search for.

    Returns:
        str: The name of the latest directory with the base name.
    """
    # List all directories in the base directory
    dirs = [
        d
        for d in os.listdir(base_dir)
        if d.startswith(base_name) and os.path.isdir(os.path.join(base_dir, d))
    ]

    # Sort directories by their timestamp to find the latest
    if not dirs:
        return None
    latest_dir = sorted(dirs, key=lambda x: x.split("_")[-1], reverse=True)[0]

    return latest_dir


def find_latest_checkpoint(output_dir, output_name):
    """
    Find the latest model and optimizer checkpoint files in the latest directory.

    Args:
        output_dir (str): The base directory to search in.
        output_name (str): The base name of the directory to search for.

    Returns:
        Tuple[str, str, int]: The paths to the latest model and optimizer files and the epoch number.
    """
    latest_dir_path = os.path.join(output_dir, output_name)

    # Check if the directory exists
    if not os.path.exists(latest_dir_path):
        raise FileNotFoundError(f"Directory {latest_dir_path} does not exist.")

    # Find all .pth files within the models directory
    model_files = [
        f
        for f in os.listdir(os.path.join(latest_dir_path, "models"))
        if f.endswith(".pth")
    ]
    if not model_files:
        raise FileNotFoundError(
            "No model or optimizer files found in the latest directory."
        )

    # Separate and sort model and optimizer files by epoch number to find the latest
    model_files_sorted = sorted(
        (f for f in model_files if "model" in f),
        key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)),
        reverse=True,
    )
    optimizer_files_sorted = sorted(
        (f for f in model_files if "optimizer" in f),
        key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)),
        reverse=True,
    )

    if not model_files_sorted or not optimizer_files_sorted:
        raise FileNotFoundError(
            "Could not find both model and optimizer files in the latest directory."
        )

    latest_model_file = model_files_sorted[0]
    latest_optimizer_file = optimizer_files_sorted[0]

    epoch_number = int(re.search(r"epoch_(\d+)", latest_model_file).group(1))

    # Construct full paths
    model_path = os.path.join(latest_dir_path, "models", latest_model_file)
    optimizer_path = os.path.join(latest_dir_path, "models", latest_optimizer_file)

    return model_path, optimizer_path, epoch_number
