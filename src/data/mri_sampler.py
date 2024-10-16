"""
Sampler for MRI data.
"""

import random

import numpy as np
import polars as pl
import torch
from fastmri.data import transforms as T

pl.Config.set_tbl_rows(50)


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
        self.seed = seed
        self._prepare_metadata()
        columns = self.metadata.columns
        self.fullysampled_column_index = columns.index("path_fullysampled")
        self.undersampled_column_index = columns.index(
            f"path_undersampled_{center_fraction}_{acceleration}"
        )
        self.slice_id_column_index = columns.index("slice_id")
        self.index_counter = 0

    def _prepare_metadata(self):
        self.metadata: pl.LazyFrame = pl.scan_csv(self.path / "metadata.csv")
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
        self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        if self.test_files:
            self.metadata = self.metadata.filter(
                pl.col("slice_id").is_in(self.test_files)
            )

        self.metadata: pl.DataFrame = self.metadata.collect()
        self.metadata = self.metadata.sample(fraction=1, seed=self.seed, shuffle=True)

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
        file_fullysampled = self.metadata[idx, self.fullysampled_column_index]
        file_undersampled = self.metadata[idx, self.undersampled_column_index]
        filename = self.metadata[idx, self.slice_id_column_index]
        self.index_counter += 1

        # Load images
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)

        return scan_fullysampled, scan_undersampled, filename

    def __len__(self):
        """Return the number of samples."""
        return len(self.metadata)
