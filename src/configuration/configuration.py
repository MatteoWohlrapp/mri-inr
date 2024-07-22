"""
Configuration module for the modulated SIREN model.
"""

import types
import yaml
import argparse

# Define the default configuration for training and testing
default_train_config = {
    "data": {
        "train": {
            "dataset": "",
            "num_samples": None,
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
        "siren_patch_size": 24,
        "activation": "sine",
    },
    "training": {
        "lr": 0.0001,
        "batch_size": 10,
        "epochs": 100,
        "output_dir": "./output",
        "output_name": "modulated_siren",
        "optimizer": "Adam",
        "logging": False,
        "criterion": "MSE",
        "model": {
            "continue_training": False,
            "model_path": None,
            "optimizer_path": None,
        },
    },
}


default_test_config = {
    "data": {
        "dataset": "",
        "test_files": None,
        "metric_samples": None,
        "visual_samples": None,
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
        "siren_patch_size": 24,
    },
    "testing": {
        "output_dir": "./output",
        "output_name": "modulated_siren",
        "model_path": "",
    },
}


def merge_configs(defaults, user_configs):
    """
    Merge the default configuration with the user configuration.

    Args:
        defaults (dict): The default configuration.
        user_configs (dict): The user configuration.

    Returns:
        dict: The merged configuration.
    """
    for key, value in user_configs.items():
        if isinstance(value, dict) and key in defaults:
            merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults


def convert_to_namespace(data):
    """
    Convert a dictionary to a namespace.

    Args:
        data (dict): The dictionary to convert.

    Returns:
        types.SimpleNamespace: The namespace.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_to_namespace(value)
        return types.SimpleNamespace(**data)
    return data


def namespace_to_dict(obj):
    """
    Convert a namespace to a dictionary.

    Args:
        obj (types.SimpleNamespace): The namespace to convert.

    Returns:
        dict: The dictionary.
    """
    if isinstance(obj, types.SimpleNamespace):
        obj = vars(obj)
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    else:
        return obj


def save_config_to_yaml(config, filename):
    """
    Save the configuration to a YAML file.

    Args:
        config (types.SimpleNamespace): The configuration to save.
        filename (str): The filename to save the configuration to.
    """
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


def load_configuration(file_path, testing=False):
    """
    Load the configuration from a YAML file.

    Args:
        file_path (str): The path to the configuration file.
        testing (bool): Whether to load the testing configuration.

    Returns:
        types.SimpleNamespace: The configuration as a namespace.
    """
    with open(file_path, "r") as file:
        user_config = yaml.safe_load(file)

    if testing:
        full_config = merge_configs(default_test_config, user_config)
    else:
        full_config = merge_configs(default_train_config, user_config)

    types_namespace = convert_to_namespace(full_config)

    return types_namespace


def load_configuration_no_defaults(file_path):
    """
    Load the configuration from a YAML file without merging with defaults.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        types.SimpleNamespace: The configuration as a namespace.
    """
    with open(file_path, "r") as file:
        user_config = yaml.safe_load(file)

    types_namespace = convert_to_namespace(user_config)

    return types_namespace


def parse_args():
    """Parse the arguments for the script."""
    parser = argparse.ArgumentParser(description="Train a modulated SIREN on MRI data.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    return parser.parse_args()
