import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable
import pathlib
import polars as pl
from src.util.tiling import (
    image_to_patches,
    filter_black_patches,
    filter_black_patches_indices,
)
import os


class MRIDataset(Dataset):
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
        output_name: str = "modulated_siren",
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
        print(data_root / "metadata.csv")
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")
        self.fullysampled_tiles: torch.Tensor = torch.empty(0)
        self.undersampled_tiles: torch.Tensor = torch.empty(0)
        self.fullysampled_column_index = None
        self.undersampled_column_index = None
        self._prepare_metadata()
        self._create_tiles()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
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
        self.slice_ids = self.metadata.select(pl.col("slice_id")).to_numpy().flatten()

        # Find the column index of the fullysampled and the specific undersampled files by looking at the columns list and checking at which index the column name is
        columns = self.metadata.columns
        self.fullysampled_column_index = columns.index("path_fullysampled")
        self.undersampled_column_index = columns.index(
            f"path_undersampled_{self.center_fraction}_{self.acceleration}"
        )

        # Print all files used for training
        files = (
            self.metadata.select(pl.col("stem").unique()).to_numpy().flatten().tolist()
        )

        files = list(set(files))
        output_dir = f"{self.output_dir}/{self.output_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/processed_files.txt", "a", encoding="utf-8") as f:
            print(f"Processing files from {self.data_root}", file=f)
            for file in files:
                print(file, file=f)

    def _create_tiles(self):
        """Split the image into tiles."""
        fullysampled_tiles = []
        undersampled_tiles = []
        for i in range(len(self.metadata)):
            file_fullysampled = self.metadata[i, self.fullysampled_column_index]
            file_undersampled = self.metadata[i, self.undersampled_column_index]
            scan_fullysampled = np.load(file_fullysampled)
            scan_undersampled = np.load(file_undersampled)
            scan_fullysampled = torch.from_numpy(scan_fullysampled)
            scan_undersampled = torch.from_numpy(scan_undersampled)
            if self.transform:
                scan_fullysampled = self.transform(scan_fullysampled)
                scan_undersampled = self.transform(scan_undersampled)

            patches, _ = image_to_patches(
                scan_fullysampled.unsqueeze(0),
                self.outer_patch_size,
                self.inner_patch_size,
            )
            fullysampled_tiles.append(patches)

            patches, _ = image_to_patches(
                scan_undersampled.unsqueeze(0),
                self.outer_patch_size,
                self.inner_patch_size,
            )
            undersampled_tiles.append(patches)

        self.undersampled_tiles, self.fullysampled_tiles = filter_black_patches(
            self.undersampled_tiles, self.fullysampled_tiles
        )
        self.fullysampled_tiles = torch.cat(fullysampled_tiles, dim=0)
        self.undersampled_tiles = torch.cat(undersampled_tiles, dim=0)

    def __len__(self):
        return self.fullysampled_tiles.shape[0]

    def __getitem__(self, idx: int):
        return self.fullysampled_tiles[idx], self.undersampled_tiles[idx]

    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]

    # TODO to be used for testing to pick a specific image for better comparison
    def get_image(self, image_slice_id: str):
        """ "
        Get a specific image from the dataset.

        Args:
            image_slice_id (str): The slice ID of the image to get.

        Returns:
            tuple: The fullysampled and the undersampled image.
        """
        idx = self.metadata.filter(pl.col("slice_id") == image_slice_id).collect()
        file_fullysampled = self.metadata[idx, self.fullysampled_column_index]
        file_undersampled = self.metadata[idx, self.undersampled_column_index]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return scan_fullysampled, scan_undersampled

    def get_random_image(self):
        """Get a random image from the dataset."""
        idx = np.random.randint(len(self.metadata))
        file_fullysampled = self.metadata[idx, self.fullysampled_column_index]
        file_undersampled = self.metadata[idx, self.undersampled_column_index]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return scan_fullysampled, scan_undersampled


class MRIDatasetLessRAM(Dataset):
    """MRIDataset class that does not load all images into memory. Use only if you have limited RAM."""

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
        output_name: str = "modulated_siren",
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
        print(data_root / "metadata.csv")
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")
        self.fullysampled_tiles: torch.Tensor = torch.empty(0)
        self.undersampled_tiles: torch.Tensor = torch.empty(0)
        self.fullysampled_column_index = None
        self.undersampled_column_index = None
        self._prepare_metadata()
        self._prepare_tiles()

    def _prepare_metadata(self):
        """Prepare the metadata for the dataset.

        This is done by creating a DataFrame that contains the metadata and paths to the relevant files.

        Returns:
            None
        """
        self.metadata = self.metadata.filter(pl.col("slice_num") <= 10)
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
        self.slice_ids = self.metadata.select(pl.col("slice_id")).to_numpy().flatten()

        # Find the column index of the fullysampled and the specific undersampled files by looking at the columns list and checking at which index the column name is
        columns = self.metadata.columns
        self.fullysampled_column_index = columns.index("path_fullysampled")
        self.undersampled_column_index = columns.index(
            f"path_undersampled_{self.center_fraction}_{self.acceleration}"
        )

        # Print all files used for training
        files = (
            self.metadata.select(pl.col("stem").unique()).to_numpy().flatten().tolist()
        )

        files = list(set(files))
        output_dir = f"{self.output_dir}/{self.output_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/processed_files.txt", "a", encoding="utf-8") as f:
            print(f"Processing files from {self.data_root}", file=f)
            for file in files:
                print(file, file=f)

    def path_to_patches(self, path: str):
        scan = np.load(path)
        scan = torch.from_numpy(scan)
        if self.transform:
            scan = self.transform(scan)
        patches, _ = image_to_patches(
            scan.unsqueeze(0),
            self.outer_patch_size,
            self.inner_patch_size,
        )
        return patches

    def _prepare_tiles(self):
        """Split the image into tiles."""
        self.undersampled_tiles: list = []
        tiles_counter = 0
        for i in range(len(self.metadata)):
            file_undersampled = self.metadata[i, self.undersampled_column_index]
            patches = self.path_to_patches(file_undersampled)
            indices = filter_black_patches_indices(patches)
            tiles_counter += len(indices)
            self.undersampled_tiles.append(
                {
                    "path": file_undersampled,
                    "indices": indices,
                    "counter": tiles_counter,
                }
            )

    def __len__(self):
        return self.undersampled_tiles[-1]["counter"]

    def __getitem__(self, idx: int):
        for i in range(len(self.undersampled_tiles)):
            if idx < self.undersampled_tiles[i]["counter"]:
                path = self.undersampled_tiles[i]["path"]
                if i == 0:
                    index = self.undersampled_tiles[i]["indices"][idx]
                else:
                    index = self.undersampled_tiles[i]["indices"][
                        idx - self.undersampled_tiles[i - 1]["counter"]
                    ]
                patches = self.path_to_patches(path)
                return patches[index]
        return None

    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]
