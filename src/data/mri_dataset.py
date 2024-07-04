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
    def __init__(self, data_root: pathlib.Path, transform: Optional[Callable] = None, number_of_samples: Optional[int] = 0, mri_type: str = 'Flair', seed: Optional[int] = 31415, specific_slice_ids: Optional[List[str]] = None):
        self.data_root: pathlib.Path = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.mri_type = mri_type
        self.seed = seed
        self.slice_ids = specific_slice_ids
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")#TODO change to parquet
        self._prepare_metadata()

    def _prepare_metadata(self):
        if self.mri_type:
            self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        if self.slice_ids:
            self.metadata = self.metadata.filter(pl.col("slice_id").is_in(self.slice_ids))
        if self.number_of_samples:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)
        else:
            self.metadata = self.metadata.collect()
        self.slice_ids = self.metadata.select(pl.col("slice_id")).to_numpy().flatten()

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

class MRIDataset2(Dataset):
    '''Improved version of the MRIDataset class
    When initialized, it loads the images and creates the patches immediately to store them in memory.
    This way, the dataset can be used with a DataLoader without having to load the images and create the patches every time.
    '''
    def __init__(self, data_root: pathlib.Path, transform: Optional[Callable] = None, number_of_samples: Optional[int] = 0, mri_type: str = 'Flair', seed: Optional[int] = 31415, specific_slice_ids: Optional[List[str]] = None):
        self.data_root: pathlib.Path = data_root
        self.transform = transform
        self.number_of_samples = number_of_samples
        self.mri_type = mri_type
        self.seed = seed
        self.slice_ids = specific_slice_ids
        self.metadata: pl.LazyFrame = pl.scan_csv(data_root / "metadata.csv")#TODO change to parquet
        self.fullysampled_tiles: torch.Tensor = torch.empty(0)
        self.undersampled_tiles: torch.Tensor = torch.empty(0)
        self._prepare_metadata()
        self._create_tiles()

    def _prepare_metadata(self):
        if self.mri_type:
            self.metadata = self.metadata.filter(pl.col("mri_type") == self.mri_type)
        if self.slice_ids:
            self.metadata = self.metadata.filter(pl.col("slice_id").is_in(self.slice_ids))
        if self.number_of_samples:
            self.metadata = self.metadata.collect().sample(n=self.number_of_samples, seed=self.seed)
        else:
            self.metadata = self.metadata.collect()
        self.slice_ids = self.metadata.select(pl.col("slice_id")).to_numpy().flatten()
    
    def _create_tiles(self):
        fullysampled_tiles = []
        undersampled_tiles = []
        for i in range(len(self.metadata)):
            file_fullysampled = self.metadata[i,0]
            file_undersampled = self.metadata[i,1]
            scan_fullysampled = np.load(file_fullysampled)
            scan_undersampled = np.load(file_undersampled)
            scan_fullysampled = torch.from_numpy(scan_fullysampled)
            scan_undersampled = torch.from_numpy(scan_undersampled)
            if self.transform:
                scan_fullysampled = self.transform(scan_fullysampled)
                scan_undersampled = self.transform(scan_undersampled)
            fullysampled_tiles.append(extract_with_inner_patches(scan_fullysampled.unsqueeze(0), 32, 32))
            undersampled_tiles.append(extract_with_inner_patches(scan_undersampled.unsqueeze(0), 32, 32))
        self.fullysampled_tiles = torch.cat(fullysampled_tiles, dim=0)
        self.undersampled_tiles = torch.cat(undersampled_tiles, dim=0)
        self.undersampled_tiles, self.fullysampled_tiles = filter_black_tiles(self.undersampled_tiles, self.fullysampled_tiles)

    def __len__(self):
        return self.fullysampled_tiles.shape[0]
    
    def __getitem__(self, idx: int):
        return self.fullysampled_tiles[idx], self.undersampled_tiles[idx]
    
    def __getitems__(self, idxs: List[int]):
        return [self.__getitem__(idx) for idx in idxs]
    
    def get_image(self, image_slice_id: str):
        idx = self.metadata.filter(pl.col("slice_id") == image_slice_id).collect().index[0]
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
    
