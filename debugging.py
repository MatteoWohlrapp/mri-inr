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

data_root = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val")



def main():
    print('start')
    process_files(data_root)
    print('end')

if __name__ == "__main__":
    main()
    

    