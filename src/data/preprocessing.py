"""
Preprocess the data
"""

import pathlib

import fastmri
import h5py
import numpy as np
import polars as pl
import torch
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from src.util.visualization import normalize_scan


def load_h5(path) -> np.ndarray:
    """
    Load k-space data from a .h5 file.

    Args:
        path (str): The path to the .h5 file.

    Returns:
        np.ndarray: The k-space data.xs
    """
    with h5py.File(path, "r") as f:
        data = f["kspace"][()]
    return data


def load_mri_scan(
    path: pathlib.Path, center_fraction, acceleration, undersampled=False
) -> torch.Tensor:
    """
    Load an MRI scan from a .h5 file.

    Args:
        path (pathlib.Path): The path to the .h5 file.
        undersampled (bool): Whether to load the undersampled scan.
        center_fraction (float): The center fraction for the mask.
        acceleration (int): The acceleration for the mask.xs

    Returns:
        torch.Tensor: The MRI scan.xs
    """
    mri_data = load_h5(path)
    mri_data = T.to_tensor(mri_data)

    if undersampled:
        mask_func = RandomMaskFunc(
            center_fractions=[center_fraction], accelerations=[acceleration]
        )
        mri_data, _, _ = T.apply_mask(mri_data, mask_func)

    mri_data = fastmri.ifft2c(mri_data)
    mri_data = fastmri.complex_abs(mri_data)
    scan = mri_data
    return scan


def get_mri_type(file: pathlib.Path) -> str:
    """
    Gets the MRI name from the filename

    Args:
        file (pathlib.Path): The file to get the MRI type from.

    Returns:
        str: The MRI type.
    """
    if "flair" in file.stem.lower():
        return "Flair"
    elif "t1" in file.stem.lower():
        return "T1"
    elif "t2" in file.stem.lower():
        return "T2"
    else:
        return None


def get_mri_area(file: pathlib.Path) -> str:
    """
    Gets the MRI area from the filename

    Args:
        file (pathlib.Path): The file to get the MRI area from.

    Returns:
        str: The MRI area.
    """
    if "brain" in file.stem.lower():
        return "Brain"
    elif "knee" in file.stem.lower():
        return "Knee"
    else:
        print(f"Unknown MRI area for file {file.stem}")
        return None


def process_files(data_root: pathlib.Path, undersample_params: list):
    """
    Process the files in the data root directory.

    Args:
        data_root (pathlib.Path): The root directory containing the data files.
        undersample_params (list): The undersample parameters to use.
    """
    metadata_keys = [
        "path_fullysampled",
        "stem",
        "slice_id",
        "slice_num",
        "width",
        "height",
        "mri_type",
        "mri_area",
    ]
    metadata_keys += [f"path_undersampled_{cf}_{acc}" for cf, acc in undersample_params]
    metadata = {key: [] for key in metadata_keys}

    dest_dir = data_root / "processed_files"
    dest_dir.mkdir(exist_ok=True)

    for file in data_root.glob("*.h5"):
        scan = normalize_scan(
            load_mri_scan(file, undersampled=False, center_fraction=1, acceleration=1)
        )
        undersampled_scans = [
            normalize_scan(
                load_mri_scan(
                    file, undersampled=True, center_fraction=cf, acceleration=acc
                )
            )
            for cf, acc in undersample_params
        ]

        for i in range(scan.shape[0]):
            metadata["path_fullysampled"].append(str(dest_dir / f"{file.stem}_{i}.npy"))
            metadata["stem"].append(file.stem)
            metadata["slice_id"].append(f"{file.stem}_{i}")
            metadata["slice_num"].append(i)
            metadata["width"].append(scan.shape[1])
            metadata["height"].append(scan.shape[2])
            metadata["mri_type"].append(get_mri_type(file))
            metadata["mri_area"].append(get_mri_area(file))

            np.save(dest_dir / f"{file.stem}_{i}.npy", scan[i].numpy())

            for (cf, acc), undersampled_scan in zip(
                undersample_params, undersampled_scans
            ):
                path_key = f"path_undersampled_{cf}_{acc}"
                metadata[path_key].append(
                    str(dest_dir / f"{file.stem}_{i}_undersampled_{cf}_{acc}.npy")
                )
                np.save(
                    dest_dir / f"{file.stem}_{i}_undersampled_{cf}_{acc}.npy",
                    undersampled_scan[i].numpy(),
                )

    metadata_df = pl.DataFrame(metadata)
    metadata_df.write_csv(dest_dir / "metadata.csv")
