"""
Sampler for MRI data.
"""

import random

import numpy as np
import polars as pl
import torch
from fastmri.data import transforms as T
import h5py
from fastmri.data.subsample import RandomMaskFunc
import fastmri
import pathlib

pl.Config.set_tbl_rows(50)


class SotaSampler:
    """Sampler for MRI data used during testing."""

    def __init__(
        self,
        data_root,
        seed=42,
        mri_type="Flair",
        test_files=None,
        transform=None,
    ):
        """
        Initialize the MRISampler.

        Args:
            path (str): The path to the data directory.
            seed (int): The seed for reproducibility.
            mri_type (str): The type of MRI scan to sample.
            test_files (list): The list of test files to sample from.
        """
        self.data_root = data_root
        self.mri_type = mri_type
        self.test_files = test_files
        self.seed = seed
        self.transform = transform  
        self._prepare_metadata()
        self.index_counter = 0

    def _prepare_metadata(self):
        self.metadata: pl.LazyFrame = pl.scan_csv(self.data_root / "sota_metadata.csv")
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
        self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        self.metadata = self.metadata.filter(pl.col("width") == 640)
        self.metadata = self.metadata.filter(pl.col("height") == 320)
        if self.test_files:
            self.metadata = self.metadata.filter(
                pl.col("slice_id").is_in(self.test_files)
            )

        self.metadata: pl.DataFrame = self.metadata.collect()
        self.metadata = self.metadata.sample(fraction=1, seed=self.seed, shuffle=True)
    
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

    def get_random_sample(self):
        """
        Get a random sample from the dataset.

        Returns:
            tuple: The fully sampled scan, the undersampled scan, and the filename.
        """
        if self.metadata.shape[0] == 0:
            raise ValueError("No samples available to select from.")

        # Randomly select an index
        idx = (
            self.index_counter % self.metadata.shape[0]
        )  # to avoid index out of bounds
        self.index_counter += 1
        row = self.metadata.row(idx)
        filename = row[2]
        kspace = str(self.data_root / row[0]) # path
        slice_idx = row[3] # slice_num
        scan_fullysampled = self.load_mri_scan(kspace)[slice_idx]
        scan_undersampled = self.load_mri_scan(kspace, 0.05, 6)[slice_idx]
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)

        scan_fullysampled = scan_fullysampled.unsqueeze(0)
        scan_undersampled = scan_undersampled.unsqueeze(0)
        return scan_fullysampled, scan_undersampled, filename

    def __len__(self):
        """Return the number of samples."""
        return len(self.metadata)
