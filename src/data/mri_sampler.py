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
        center_fraction: float = 0.05,
        acceleration: int = 6,
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
        self._prepare_metadata()
        self.indices = list(range(len(self.metadata)))
        columns = self.metadata.columns
        self.fullysampled_column_index = columns.index("path_fullysampled")
        self.undersampled_column_index = columns.index(
            f"path_undersampled_{center_fraction}_{acceleration}"
        )
        self.slice_id_column_index = columns.index("slice_id")

    def _prepare_metadata(self):
        self.metadata: pl.LazyFrame = pl.scan_csv(self.path / "metadata.csv")
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
        self.metadata = pl.scan_csv(self.path / "metadata.csv").filter(
            pl.col("mri_type") == self.mri_type
        )
        if self.test_files:
            self.metadata = self.metadata.filter(pl.col("stem").is_in(self.test_files))
        self.metadata = self.metadata.collect()

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
        file_fullysampled = self.metadata[idx, self.fullysampled_column_index]
        file_undersampled = self.metadata[idx, self.undersampled_column_index]
        filename = self.metadata[idx, self.slice_id_column_index]

        # Load images
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)

        return scan_fullysampled, scan_undersampled, filename

    def __len__(self):
        """Return the number of samples."""
        return len(self.metadata)
