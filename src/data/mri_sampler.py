"""
Sampler for MRI data.
"""

import numpy as np
import random
from fastmri.data import transforms as T
import polars as pl
import torch


class MRISampler:
    """Sampler for MRI data used during testing."""

    def __init__(
        self,
        path,
        seed=42,
        mri_type="Flair",
        test_files=None,
    ):
        """
        Initialize the MRISampler.

        Args:
            path (str): The path to the data directory.
            seed (int): The seed for reproducibility.
            mri_type (str): The type of MRI scan to sample.
            test_files (list): The list of test files to sample from.
        """
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
        """
        Get a random sample from the dataset.

        Returns:
            tuple: The fully sampled scan, the undersampled scan, and the filename.
        """
        if not self.indices:
            raise ValueError("No samples available to select from.")

        # Randomly select an index
        idx = random.choice(self.indices)
        file_fullysampled = self.metadata[idx, 0]
        file_undersampled = self.metadata[idx, 8]
        filename = self.metadata[idx, 2]

        # Load images
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)

        return scan_fullysampled, scan_undersampled, filename
