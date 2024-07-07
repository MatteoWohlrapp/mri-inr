import pathlib
from src.data.preprocessing import process_file

def run_preprocessing(path):
    # Call the process_file function from the preprocessing module
    process_file(path)

if __name__ == "__main__":
    path  = pathlib.Path("data/raw")
    # Call the run_preprocessing function with the provided path
    run_preprocessing(path)