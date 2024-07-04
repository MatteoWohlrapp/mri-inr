import pathlib
import h5py
import numpy as np
import torch
import polars as pl
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc



def load_h5(path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        data = f["kspace"][()]
    return data

def load_mri_scan(path: pathlib.Path, undersampled=False) -> torch.Tensor:
    mri_data = load_h5(path)
    mri_data = T.to_tensor(mri_data)

    if undersampled:
        mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
        mri_data, _, _ = T.apply_mask(mri_data, mask_func)

    mri_data = fastmri.ifft2c(mri_data)
    mri_data = fastmri.complex_abs(mri_data)
    scan = mri_data
    return scan

def test_kspace(path: pathlib.Path, slice_num = 0) -> torch.Tensor:
    mri_data = load_h5(path)
    mri_data = T.to_tensor(mri_data)

    mask_func1 = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
    mask_func2 = RandomMaskFunc(center_fractions=[0.1], accelerations=[4])
    mask_func3 = RandomMaskFunc(center_fractions=[0.9], accelerations=[8])
    mask_func4 = RandomMaskFunc(center_fractions=[0.9], accelerations=[4])

    mri_data1, _, _ = T.apply_mask(mri_data, mask_func1)
    mri_data2, _, _ = T.apply_mask(mri_data, mask_func2)
    mri_data3, _, _ = T.apply_mask(mri_data, mask_func3)
    mri_data4, _, _ = T.apply_mask(mri_data, mask_func4)

    mri_data1 = fastmri.complex_abs(mri_data1)
    mri_data2 = fastmri.complex_abs(mri_data2)
    mri_data3 = fastmri.complex_abs(mri_data3)
    mri_data4 = fastmri.complex_abs(mri_data4)
    return [mri_data1[slice_num], mri_data2[slice_num], mri_data3[slice_num], mri_data4[slice_num]]

def normalize_scan(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

def get_mri_type(file: pathlib.Path) -> str:
    if "flair" in file.stem.lower():
        return "Flair"
    elif "t1" in file.stem.lower():
        return "T1"
    elif "t2" in file.stem.lower():
        return "T2"
    else:
        return None


def process_files(data_root: pathlib.Path):
    """Alternative implementation of process_files that only creates one metadata entry per slice by having path_undersampled and path_fullysampled columns."""
    metadata = {
        "path_fullysampled": [],
        "path_undersampled": [],
        "stem": [],
        "slice_id": [], # stem + slice_id = unique identifier
        "slice_num": [],
        "width": [],
        "height": [],
        "mri_type": [],
    }
    dest_dir = data_root / "processed"
    dest_dir.mkdir(exist_ok=True)
    for file in data_root.glob("*.h5"):
        scan = load_mri_scan(file, undersampled=False)
        undersampled_scan = load_mri_scan(file, undersampled=True)
        scan = normalize_scan(scan)
        undersampled_scan = normalize_scan(undersampled_scan)
        for i in range(scan.shape[0]):
            metadata["path_fullysampled"].append(str(dest_dir / f"{file.stem}_{i}.npy"))
            metadata["path_undersampled"].append(str(dest_dir / f"{file.stem}_{i}_undersampled.npy"))
            metadata["stem"].append(file.stem)
            metadata["slice_id"].append(f"{file.stem}_{i}")
            metadata["slice_num"].append(i)
            metadata["width"].append(scan.shape[1])
            metadata["height"].append(scan.shape[2])
            metadata["mri_type"].append(get_mri_type(file))
            np.save(dest_dir / f"{file.stem}_{i}.npy", scan[i].numpy())
            np.save(dest_dir / f"{file.stem}_{i}_undersampled.npy", undersampled_scan[i].numpy())
    metadata = pl.DataFrame(metadata)
    print(metadata)
    print(data_root)
    metadata.write_csv(dest_dir / "metadata.csv")