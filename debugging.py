import pathlib
from src.data.preprocessing import process_files
from src.util.visualization import metrics_density_plot, metrics_boxplot
import numpy as np


def test_plotting():
    metrics = {
        "metric1": np.random.standard_normal(1000),
        "metric2": np.random.rand(1000),
    }
    output_dir = r"./output"
    metrics_density_plot(metrics, output_dir)
    metrics_boxplot(metrics, output_dir)

if __name__ == "__main__":
    test_plotting()
