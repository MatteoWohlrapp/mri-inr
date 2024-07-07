import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable
import pathlib
import polars as pl
from src.util.tiling import (
    image_to_patches,
    filter_black_tiles,
)
import os


class MRIDataset(Dataset):
    """Improved version of the MRIDataset class
    When initialized, it loads the images and creates the patches immediately to store them in memory.
    This way, the dataset can be used with a DataLoader without having to load the images and create the patches every time.
    """

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
    ):
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
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")
        self.fullysampled_tiles: torch.Tensor = torch.empty(0)
        self.undersampled_tiles: torch.Tensor = torch.empty(0)
        self._prepare_metadata()
        self._create_tiles()

    def _prepare_metadata(self):
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

        # Print all files used for training
        files = []
        for i in range(len(self.metadata)):
            files.append(self.metadata[i, 2])

        files = list(set(files))
        output_dir = f"{self.output_dir}/{self.output_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/processed_files.txt", "w", encoding="utf-8") as f:
            for file in files:
                print(file, file=f)

    def _create_tiles(self):
        fullysampled_tiles = []
        undersampled_tiles = []
        for i in range(len(self.metadata)):
            file_fullysampled = self.metadata[i, 0]
            file_undersampled = self.metadata[i, 1]
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

        self.fullysampled_tiles = torch.cat(fullysampled_tiles, dim=0)
        self.undersampled_tiles = torch.cat(undersampled_tiles, dim=0)
        self.undersampled_tiles, self.fullysampled_tiles = filter_black_tiles(
            self.undersampled_tiles, self.fullysampled_tiles
        )

    def __len__(self):
        return self.fullysampled_tiles.shape[0]

    def __getitem__(self, idx: int):
        return self.fullysampled_tiles[idx], self.undersampled_tiles[idx]

    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]

    def get_image(self, image_slice_id: str):
        idx = (
            self.metadata.filter(pl.col("slice_id") == image_slice_id)
            .collect()
            .index[0]
        )
        file_fullysampled = self.metadata[idx, 0]
        file_undersampled = self.metadata[idx, 1]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return scan_fullysampled, scan_undersampled

    def get_random_image(self):
        idx = np.random.randint(len(self.metadata))
        file_fullysampled = self.metadata[idx, 0]
        file_undersampled = self.metadata[idx, 1]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return scan_fullysampled, scan_undersampled
