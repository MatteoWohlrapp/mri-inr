"""
This script trains a SOTA models for MRI reconstruction.
"""

import datetime
import os
import pathlib

import torch

from src.configuration.configuration import (
    load_configuration,
    namespace_to_dict,
    parse_args,
    save_config_to_yaml,
)
from src.data.sota_dataset import SOTADataset
from src.networks.unet.unet import UNet
from src.train.training import Trainer
from src.util.slurm_restart import find_latest_checkpoint, find_latest_folder
from src.util.timing import time_function
import torchvision.transforms as transforms
from src.util.visualization import normalize_scan


@time_function
def train_sota(config):
    """
    Train the modulated SIREN model.

    Args:
        config (argparse.Namespace): The configuration to use for training.
    """
    continue_training = False
    # Checking if we want to continue training
    if config.training.model.continue_training and not config.training.model.model_path:
        print("Continue Training of SOTA.")
        output_name = find_latest_folder(
            config.training.output_dir, config.training.output_name
        )

        if output_name:
            config.training.output_name = output_name
            continue_training = True
    elif config.training.model.continue_training:
        continue_training = True

    if not continue_training:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.training.output_name = f"{config.training.output_name}_{current_time}"

    # Write configuration to file
    config_path = pathlib.Path(config.training.output_dir) / config.training.output_name
    os.makedirs(config_path, exist_ok=True)
    with open(f"{config_path}/config.yaml", "w", encoding="utf-8") as f:
        dict = namespace_to_dict(config)
        save_config_to_yaml(dict, f"{config_path}/config.yaml")

    transform = transforms.Compose(
        [
            normalize_scan,
        ]
    )

    # Load dataset
    train_dataset = SOTADataset(
        pathlib.Path(config.data.train.dataset),
        number_of_samples=config.data.train.num_samples,
        outer_patch_size=config.model.outer_patch_size,
        inner_patch_size=config.model.inner_patch_size,
        output_dir=config.training.output_dir,
        output_name=config.training.output_name,
        acceleration=config.data.acceleration,
        center_fraction=config.data.center_fraction,
        transform=transform,
    )

    val_dataset = (
        SOTADataset(
            pathlib.Path(config.data.val.dataset),
            number_of_samples=config.data.val.num_samples,
            outer_patch_size=config.model.outer_patch_size,
            inner_patch_size=config.model.inner_patch_size,
            output_dir=config.training.output_dir,
            output_name=config.training.output_name,
            acceleration=config.data.acceleration,
            center_fraction=config.data.center_fraction,
            transform=transform,
        )
        if config.data.val.dataset
        else None
    )

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model
    unet = UNet()

    unet.to(device)

    # Define the criterion and optimizer
    if config.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(unet.parameters(), lr=config.training.lr)
    elif config.training.optimizer == "SGD":
        optimizer = torch.optim.SGD(unet.parameters(), lr=config.training.lr)
    else:
        raise ValueError("Unsupported optimizer")

    # Define the trainer
    trainer = Trainer(
        model=unet,
        device=device,
        train_dataset=train_dataset,
        optimizer=optimizer,
        output_dir=config.training.output_dir,
        val_dataset=val_dataset,
        batch_size=config.training.batch_size,
        output_name=config.training.output_name,
        outer_patch_size=0,
        inner_patch_size=0,
        siren_patch_size=0,
        save_interval=config.training.save_interval,
        num_workers=config.data.train.num_workers,
        logging=config.training.logging,
        criterion=config.training.criterion,
    )

    initial_epoch = 0

    # Check if we want to load an existing model
    if continue_training:
        if config.training.model.model_path:
            trainer.load_model(
                model_path=config.training.model.model_path,
                optimizer_path=config.training.model.optimizer_path,
            )
        else:
            model_path, optimizer_path, epoch_number = find_latest_checkpoint(
                config.training.output_dir, config.training.output_name
            )
            trainer.load_model(
                model_path=pathlib.Path(model_path),
                optimizer_path=pathlib.Path(optimizer_path),
            )
            initial_epoch = epoch_number + 1

    print(
        f"Training the modulated SIREN..., output name is {config.training.output_name}", flush=True
    )

    # Train the model
    trainer.train(initial_epoch, config.training.epochs)


if __name__ == "__main__":
    args = parse_args()
    config_path = pathlib.Path(args.config)
    config = load_configuration(config_path)
    train_sota(config)
