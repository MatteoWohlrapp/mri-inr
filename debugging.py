import pathlib

import numpy as np
import polars as pl
import torch

from src.data.preprocessing import process_files, test_kspace
from src.data.mri_dataset import MRIDataset
from src.util.visualization import show_image, show_image_comparison, show_batch, show_images, save_image
from einops import rearrange
from src.configuration.simple_conf import load_configuration
from src.util.tiling import extract_with_inner_patches, classify_tile, collate_fn, extract_center
from src.data.mri_dataset import MRIDataset2, MRIDataset

data_root = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val")

def test_mr_dataset2():
    dataset = MRIDataset(data_root)
    print(len(dataset))
    print(type(dataset[0][0]))
    print(dataset[0][0].shape)
    dataset2 = MRIDataset2(data_root)
    print(len(dataset2))
    print(type(dataset2[0][0]))
    print(dataset2[0][0].shape)

def print_k_space():
    path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\file_brain_AXFLAIR_200_6002425.h5")
    scans = [np.log(scan+ 1e-9)  for scan in test_kspace(path)]
    labels = ["center_fraction=0.1, acceleration=8", "center_fraction=0.1, acceleration=4", "center_fraction=0.9, acceleration=8", "center_fraction=0.9, acceleration=4"]
    #show_images(scans, labels=labels)
    save_image(scans[0], path = pathlib.Path(r'C:\Users\jan\k_space.png'))

def main():
    print('start')
    test_mr_dataset2()
    print('end')

if __name__ == "__main__":
    main()
    

    