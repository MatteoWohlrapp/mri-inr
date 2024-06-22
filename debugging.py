import pathlib

import numpy as np
import polars as pl
import torch

from src.data.preprocessing import process_files
from src.data.mri_dataset import MRIDataset
from src.util.visualization import show_image, show_image_comparison, show_batch
from einops import rearrange
from src.configuration.simple_conf import load_configuration
from src.util.tiling import extract_with_inner_patches, classify_tile, collate_fn, extract_center

data_root = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train")

def load_pt_file(path: pathlib.Path):
    return torch.load(path)

def test_smth():
    tensors = [
            torch.linspace(-1, 1, steps=5), #TODO make this dynamic
            torch.linspace(-1, 1, steps=5), #TODO make this dynamic
        ]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = rearrange(mgrid, "h w b -> (h w) b")
    print(mgrid.clone().detach().repeat(3, 1, 1).requires_grad_())

def test1():
    dataset = MRIDataset(data_root, transform=None, number_of_samples=15, mri_type='Flair')
    img = dataset[1][0].unsqueeze(0)
    tiles = extract_with_inner_patches(img, 32, 16)
    # iterate over tiles and set them to 1 if they are just black
    black_count = 0
    for i, tile in enumerate(tiles):
        if classify_tile(tile) == 0:
            tiles[i] = torch.ones_like(tile)
            black_count += 1
    print(f"Black tiles: {black_count}")
    tiles_subset = tiles[0:200]
    show_batch(tiles_subset, ncols=10)

def test2():
    dataset = MRIDataset(data_root, transform=None, number_of_samples=15, mri_type='Flair')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(data_loader))[0]
    show_batch(batch, ncols=10)

def test3():
    # tensor 6x6
    tensor = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35,36]])
    print(tensor)
    # extract center 4x4
    center = extract_center(tensor, 6,2)
    print(center)

if __name__ == "__main__":
    test3()
    

    