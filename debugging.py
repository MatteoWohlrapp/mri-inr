import pathlib
from src.data.preprocessing import process_files
from src.util.visualization import metrics_density_plot, metrics_boxplot, save_image
from src.data.mri_sampler import MRISampler
import numpy as np


def plot_images(n = 10):
    # Just create plots of n fully sampled images.
    input_dir = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train\processed_files")
    output_dir = pathlib.Path(r"./output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = MRISampler(input_dir)
    for i in range(n):
        print(i)
        image ,_ , slice_id =  sampler.get_random_sample()
        print(slice_id)
        save_image(image, slice_id, output_dir)

    

def test_plotting():
    metrics = {
        "metric1": np.random.rand(1000),
        "metric2": np.random.rand(1000),
    }
    output_dir = r"./output"
    metrics_density_plot(metrics, output_dir)
    metrics_boxplot(metrics, output_dir)

if __name__ == "__main__":
    plot_images()
