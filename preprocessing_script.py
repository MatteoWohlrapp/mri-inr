import argparse
import pathlib
from src.data.preprocessing import process_files


def run_preprocessing(path: pathlib.Path, mask_params: list):
    """
    Preprocesses the MRI files in the given path.

    Args:
    path (pathlib.Path): Path to the data directory.
    mask_params (list): List of tuples, each containing the threshold and kernel size for masking.
    """
    process_files(path, mask_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some MRI files.")
    parser.add_argument(
        "-p",
        "--path",
        type=pathlib.Path,
        required=True,
        help="Path to the data directory.",
    )
    args = parser.parse_args()
    path = args.path
    mask_params = [
        (0.05, 6),
        (0.1, 6),
    ]  # You can also take these from command line if needed
    run_preprocessing(path, mask_params)
