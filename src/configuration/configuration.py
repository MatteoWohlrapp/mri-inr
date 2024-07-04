import types
import yaml
import argparse



def convert_to_namespace(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_to_namespace(value)
        return types.SimpleNamespace(**data)
    return data

def load_configuration(file_path):
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    types_namespace = convert_to_namespace(config_data)
    return types_namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Train a modulated SIREN on MRI data.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()