"""
Preprocess the data
"""

import pathlib
import random

import fastmri
import h5py
import numpy as np
import polars as pl
import torch
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from src.util.visualization import normalize_scan

def apply_cross_mask(data):
    center_width = 8
    mask = np.zeros_like(data)
    width, height = data.shape[2], data.shape[1]
    mask[:,:, width//2-center_width:width//2+center_width,:] = 1
    mask[:, height//2-center_width:height//2+center_width,:,:] = 1
    for _ in range(40):
        x = random.randint(0, width-1)
        mask[:,:,x,:] = 1
        y = random.randint(0, height-1)
        mask[:,y,:,:] = 1
    print(np.sum(mask)/(np.sum(np.ones_like(mask))))
    return data * mask

def apply_cricular_mask(data):
    mask = np.zeros_like(data)
    width, height = data.shape[2], data.shape[1]
    for i in range(width):
        for j in range(height):
            if (i-width//2)**2 + (j-height//2)**2 < 150:
                mask[:,j,i,:] = 1

    # add circles with random radius around the center
    for _ in range(40):
        x =  width//2
        y = height//2
        thickness = 500
        r = random.randint(0, 320*320/4)
        for i in range(width):
            for j in range(height):
                if r-thickness < (i-x)**2 + (j-y)**2 < r:
                    mask[:,j,i,:] = 1
    print(np.sum(mask)/(np.sum(np.ones_like(mask))))
    return data * mask
    

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


def load_mri_scan2(
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
        #mri_data, _, _ = T.apply_mask(mri_data, mask_func)
        mri_data = apply_cricular_mask(mri_data)
    k_space = torch.log(torch.abs(fastmri.complex_abs(mri_data)+ 1e-9))
    mri_data = fastmri.ifft2c(mri_data)
    mri_data = fastmri.complex_abs(mri_data)
    scan = mri_data
    return scan, k_space

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
    k_space = torch.log(torch.abs(fastmri.complex_abs(mri_data)+ 1e-9))
    mri_data = fastmri.ifft2c(mri_data)
    mri_data = fastmri.complex_abs(mri_data)
    scan = mri_data
    return scan, k_space

def load_mri_scan_complex(
    path: pathlib.Path, center_fraction, acceleration, undersampled=False
) -> torch.Tensor:
    """
    Load an MRI scan from a .h5 file. Without using absolute value but using the real and imaginary parts.

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
    mri_data_real = torch.abs(mri_data[:,:,:,0])
    mri_data_imag = torch.abs(mri_data[:,:,:,1])
    return mri_data_real, mri_data_imag

def load_mri_scan_complex_comb(
    path: pathlib.Path, center_fraction, acceleration, undersampled=False
) -> torch.Tensor:
    """
    Load an MRI scan from a .h5 file. Without using absolute value but using the real and imaginary parts.

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
    return mri_data


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

    dest_dir = data_root / "processed_files_complex"
    dest_dir.mkdir(exist_ok=True)

    for file in data_root.glob("*.h5"):
        scan = normalize_scan(
            load_mri_scan_complex_comb(file, undersampled=False, center_fraction=1, acceleration=1)
        )
        undersampled_scans = [
            normalize_scan(
                load_mri_scan_complex_comb(
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
