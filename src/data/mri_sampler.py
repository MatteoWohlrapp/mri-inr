import os
import h5py
import numpy as np
import random
from fastmri.data.subsample import RandomMaskFunc
import fastmri
from fastmri.data import transforms as T
import polars as pl
import torch


class MRISampler:
    def __init__(
        self,
        path,
        seed=42,
        mri_type="Flair",
        test_files=None,
    ):
        self.path = path
        self.mri_type = mri_type
        self.test_files = test_files
        random.seed(seed)  # Seed for reproducibility
        self.metadata: pl.LazyFrame = pl.scan_csv(path / "metadata.csv")
        self.metadata = pl.scan_csv(path / "metadata.csv").filter(
            pl.col("mri_type") == self.mri_type
        )
        if self.test_files:
            self.metadata = self.metadata.filter(pl.col("stem").is_in(self.test_files))
        self.metadata = self.metadata.collect()
        self.indices = list(range(len(self.metadata)))

    def get_random_sample(self):
        if not self.indices:
            raise ValueError("No samples available to select from.")

        # Randomly select an index
        idx = random.choice(self.indices)
        file_fullysampled = self.metadata[idx, 0]
        file_undersampled = self.metadata[idx, 1]
        filename = self.metadata[idx, 2]

        # Load images
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)

        return scan_fullysampled, scan_undersampled, filename
