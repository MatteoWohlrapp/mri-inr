import types
import yaml
import argparse
import datetime

default_train_config = {
    "data": {
        "train": {
            "dataset": "",
            "num_samples": 100,
            "mri_type": "FLAIR",
            "num_workers": 4,
        },
        "val": {
            "dataset": None,
            "num_samples": 10,
            "mri_type": "FLAIR",
            "num_workers": 4,
        },
    },
    "model": {
        "dim_in": 2,
        "dim_hidden": 256,
        "dim_out": 1,
        "latent_dim": 256,
        "num_layers": 5,
        "w0": 1.0,
        "w0_initial": 30.0,
        "use_bias": True,
        "dropout": 0.1,
        "encoder_type": "default",
        "encoder_path": "./model/custom_encoder.pth",
        "outer_patch_size": 32,
        "inner_patch_size": 16,
    },
    "training": {
        "lr": 0.0001,
        "batch_size": 10,
        "epochs": 100,
        "limit_io": False,
        "output_dir": "./output",
        "output_name": "modulated_siren",
        "optimizer": "Adam",
        "model": {"continue_training": False, "model_path": "", "optimizer_path": ""},
    },
}


default_test_config = {
    "data": {"dataset": "", "num_samples": 100, "test_files": None},
    "model": {
        "dim_in": 2,
        "dim_hidden": 256,
        "dim_out": 1,
        "latent_dim": 256,
        "num_layers": 5,
        "w0": 1.0,
        "w0_initial": 30.0,
        "use_bias": True,
        "dropout": 0.1,
        "encoder_type": "default",
        "encoder_path": "./model/custom_encoder.pth",
        "outer_patch_size": 32,
        "inner_patch_size": 16,
    },
    "testing": {
        "output_dir": "./output",
        "output_name": "modulated_siren",
        "model_path": "",
    },
}


def merge_configs(defaults, user_configs):
    for key, value in user_configs.items():
        if isinstance(value, dict) and key in defaults:
            merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults


def convert_to_namespace(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_to_namespace(value)
        return types.SimpleNamespace(**data)
    return data


def load_configuration(file_path, testing=False):
    with open(file_path, "r") as file:
        user_config = yaml.safe_load(file)

    if testing:
        full_config = merge_configs(default_test_config, user_config)
    else:
        full_config = merge_configs(default_train_config, user_config)

    types_namespace = convert_to_namespace(full_config)

    if not testing:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        types_namespace.training.output_name = (
            f"{types_namespace.training.output_name}_{current_time}"
        )

    return types_namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Train a modulated SIREN on MRI data.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    return parser.parse_args()
