# TODO remove from final version for submission
import pathlib
from src.data.preprocessing import process_files
from src.util.visualization import metrics_density_plot, metrics_boxplot
import numpy as np
from src.data.mri_dataset import MRIDatasetLessRAM


def test_low_ram_dataset(path: pathlib.Path):
    dataset = MRIDatasetLessRAM(path, number_of_samples=10)
    for i in range(len(dataset)):
        print(i)
        sample = dataset[i]
        print(sample)

if __name__ == "__main__":
    path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\processed_files")
    test_low_ram_dataset(path)

