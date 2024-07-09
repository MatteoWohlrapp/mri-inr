import pathlib
from src.data.preprocessing import process_files

def run_preprocessing(path: pathlib.Path, mask_params: list):
    # Call the process_file function from the preprocessing module
    process_files(path, mask_params)

if __name__ == "__main__":
    path  = pathlib.Path(r"../../dataset/fastmri/brain/singlecoil_train/")
    mask_params = [(0.05, 6), (0.1, 6)]

    # Call the run_preprocessing function with the provided path
    run_preprocessing(path, mask_params)