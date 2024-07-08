import pathlib
from src.data.preprocessing import process_files

def run_preprocessing(path):
    # Call the process_file function from the preprocessing module
    process_files(path)

if __name__ == "__main__":
    path  = pathlib.Path(r"../../dataset/fastmri/brain/singlecoil_train/")
    # Call the run_preprocessing function with the provided path
    run_preprocessing(path)