import pathlib
from src.data.preprocessing import process_files


def run_preprocessing(path: pathlib.Path, mask_params: list):
    process_files(path, mask_params)


if __name__ == "__main__":
    path = pathlib.Path(r"../../dataset/fastmri/brain/singlecoil_val/")
    mask_params = [(0.05, 6), (0.1, 6)]
    run_preprocessing(path, mask_params)