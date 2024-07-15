import pathlib
from src.data.preprocessing import process_files
from src.networks.encoding.custom_mri_encoder import test_autoencoder
import types
from src.configuration.configuration import convert_to_namespace

def run_preprocessing(path: pathlib.Path, mask_params: list):
    # Call the process_file function from the preprocessing module
    process_files(path, mask_params)

if __name__ == "__main__":
    path  = pathlib.Path(r"../../dataset/fastmri/brain/singlecoil_val/processed_v2")
        
    test_config = {
    "data": {
        "dataset": r"../../dataset/fastmri/brain/singlecoil_val",
        "test_files": None,
        "num_samples": 10,
    },
    "model": {
        "outer_patch_size": 32,
        "inner_patch_size": 16,
        "siren_patch_size": 32,
    },
    "testing": {
        "model_path": r"../models/20240530-170738_autoencoder_v1_256.pth",
        "output_dir": "output",
        "output_name": "test",
    },
}

    test_config = convert_to_namespace(test_config)
    test_autoencoder(test_config)