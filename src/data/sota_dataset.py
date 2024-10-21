import os
import pathlib
from typing import Callable, List, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
import h5py
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
import fastmri

from src.util.tiling import (
    filter_black_patches,
    filter_black_patches_indices,
    image_to_patches,
)


class SOTADataset(Dataset):
    """MRIDataset class."""

    def __init__(
        self,
        data_root: pathlib.Path,
        transform: Optional[Callable] = None,
        number_of_samples: Optional[int] = 0,
        mri_type: str = "Flair",
        seed: Optional[int] = 31415,
        specific_slice_ids: Optional[List[str]] = None,
        outer_patch_size: int = 32,
        inner_patch_size: int = 16,
        output_dir: str = "output",
        output_name: str = "sota",
        center_fraction: float = 0.05,
        acceleration: int = 6,
    ):
        """
        Initialize the MRIDataset.

        Args:
            data_root (pathlib.Path): The path to the data directory.
            transform (Optional[Callable]): The transform to apply to the data.
            number_of_samples (Optional[int]): The number of samples to use.
            mri_type (str): The type of MRI scan to use.
            seed (Optional[int]): The seed for reproducibility.
            specific_slice_ids (Optional[List[str]]): The specific slice IDs to use.
            outer_patch_size (int): The size of the outer patch.
            inner_patch_size (int): The size of the inner patch.
            output_dir (str): The output directory.
            output_name (str): The output name.
            center_fraction (float): The center fraction for the mask.
            acceleration (int): The acceleration for the mask.
        """
        self.data_root: pathlib.Path = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.mri_type = mri_type
        self.seed = seed
        self.slice_ids = specific_slice_ids
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.output_dir = output_dir
        self.output_name = output_name
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "sota_metadata.csv")
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
        self.metadata = self.metadata.filter(pl.col("width") == 640)
        self.metadata = self.metadata.filter(pl.col("height") == 320)
        if self.mri_type:
            self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        if self.slice_ids:
            self.metadata = self.metadata.filter(
                pl.col("slice_id").is_in(self.slice_ids)
            )
        if self.number_of_samples:
            self.metadata = self.metadata.collect().sample(
                n=self.number_of_samples, seed=self.seed
            )
        else:
            self.metadata = self.metadata.collect()

        # Print all files used for training
        files = (
            self.metadata.select(pl.col("file_name").unique()).to_numpy().flatten().tolist()
        )

        files = list(set(files))
        output_dir = f"{self.output_dir}/{self.output_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/processed_files.txt", "a", encoding="utf-8") as f:
            print(f"Processing files from {self.data_root}", file=f)
            for file in files:
                print(file, file=f)

    def load_h5(self, path) -> np.ndarray:
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
        self, path: pathlib.Path, center_fraction = None, acceleration = None
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
        mri_data = self.load_h5(path)
        mri_data = T.to_tensor(mri_data)

        if center_fraction is not None and acceleration is not None:
            mask_func = RandomMaskFunc(
                center_fractions=[center_fraction], accelerations=[acceleration]
            )
            mri_data, _, _ = T.apply_mask(mri_data, mask_func)

        mri_data = fastmri.ifft2c(mri_data)
        mri_data = fastmri.complex_abs(mri_data)
        scan = mri_data
        return scan

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.row(idx)
        kspace = str(self.data_root / row[0]) # path
        slice_idx = row[3] # slice_num
        scan_fullysampled = self.load_mri_scan(kspace)[slice_idx]
        scan_undersampled = self.load_mri_scan(kspace, self.center_fraction, self.acceleration)[slice_idx]
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)

        scan_fullysampled = scan_fullysampled.unsqueeze(0)
        scan_undersampled = scan_undersampled.unsqueeze(0)
        return scan_fullysampled, scan_undersampled

    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]

    def get_image(self, image_slice_id: str):
        """ "
        Get a specific image from the dataset.

        Args:
            image_slice_id (str): The slice ID of the image to get.

        Returns:
            tuple: The fullysampled and the undersampled image.
        """
        idx = self.metadata.filter(pl.col("slice_id") == image_slice_id).collect()
        row = self.metadata.row(idx)
        kspace = str(self.data_root / row[0]) 
        slice_idx = row[3]
        scan_fullysampled = self.load_mri_scan(kspace)[slice_idx]
        scan_undersampled = self.load_mri_scan(kspace, self.center_fraction, self.acceleration)[slice_idx]
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        scan_fullysampled = scan_fullysampled.unsqueeze(0)
        scan_undersampled = scan_undersampled.unsqueeze(0)
        return scan_fullysampled, scan_undersampled

    def get_random_image(self):
        """Get a random image from the dataset."""
        idx = np.random.randint(len(self.metadata))
        row = self.metadata.row(idx)
        kspace = str(self.data_root / row[0]) 
        slice_idx = row[3]
        scan_fullysampled = self.load_mri_scan(kspace)[slice_idx]
        scan_undersampled = self.load_mri_scan(kspace, self.center_fraction, self.acceleration)[slice_idx]
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        scan_fullysampled = scan_fullysampled.unsqueeze(0)
        scan_undersampled = scan_undersampled.unsqueeze(0)
        return scan_fullysampled, scan_undersampled
