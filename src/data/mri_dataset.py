import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
import pathlib
import polars as pl
from src.util.tiling import extract_with_inner_patches, classify_tile, filter_black_tiles





class MRIDataset(Dataset):
    def __init__(self, data_root: pathlib.Path, transform: Optional[Callable] = None, number_of_samples: Optional[int] = None, mri_type: str = 'Flair', seed: Optional[int] = 31415):
        self.data_root: pathlib.Path = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.mri_type = mri_type
        self.seed = seed
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")#TODO change to parquet
        self._prepare_metadata()

    def _prepare_metadata(self):
        if self.mri_type:
            self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        if self.number_of_samples:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int):
        return self._getitem_tiles(idx)


    def _getitem_img(self, idx: int):
        file_fullysampled = self.metadata[idx,0]
        file_undersampled = self.metadata[idx,1]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return scan_fullysampled, scan_undersampled
    
    def _getitem_tiles(self, idx: int):
        file_fullysampled = self.metadata[idx,0]
        file_undersampled = self.metadata[idx,1]
        scan_fullysampled = np.load(file_fullysampled)
        scan_undersampled = np.load(file_undersampled)
        scan_fullysampled = torch.from_numpy(scan_fullysampled)
        scan_undersampled = torch.from_numpy(scan_undersampled)
        if self.transform:
            scan_fullysampled = self.transform(scan_fullysampled)
            scan_undersampled = self.transform(scan_undersampled)
        return filter_black_tiles(extract_with_inner_patches(scan_fullysampled.unsqueeze(0), 32, 32), extract_with_inner_patches(scan_undersampled.unsqueeze(0), 32, 32))
    
    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]
