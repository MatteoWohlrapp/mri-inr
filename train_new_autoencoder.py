import datetime
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.configuration.configuration import load_configuration_no_defaults
from src.data.mri_dataset import MRIDataset
from src.networks.encoding.new_encoder import (
    Autoencoder_v1,
    Autoencoder_v2,
    Trainer,
    HardcodedAutoencoder,
)

config = load_configuration_no_defaults(r"./src/configuration/train_autoencoder.yaml")


def train_autoencoder(args):
    """
    Train the autoencoder.

    Args:
        args (argparse.Namespace): The arguments to use for training.
    """
    print("Training the autoencoder...", flush=True)
    print(config)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = pathlib.Path(args.training.output_dir) / current_time
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset = MRIDataset(
        pathlib.Path(args.data.train.dataset),
        number_of_samples=args.data.train.num_samples,
        outer_patch_size=args.model.outer_patch_size,
        inner_patch_size=args.model.inner_patch_size,
        output_dir=args.training.output_dir,
        acceleration=args.data.acceleration,
        center_fraction=args.data.center_fraction,
    )
    print(len(train_dataset))
    val_dataset = MRIDataset(
        pathlib.Path(args.data.val.dataset),
        number_of_samples=args.data.val.num_samples,
        outer_patch_size=args.model.outer_patch_size,
        inner_patch_size=args.model.inner_patch_size,
        output_dir=args.training.output_dir,
        acceleration=args.data.acceleration,
        center_fraction=args.data.center_fraction,
    )

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using the GPU")
    else:
        device = torch.device("cpu")

    # Load the model
    autoencoder = HardcodedAutoencoder()
    autoencoder.to(device)

    # Define the criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.training.lr)

    # Define the trainer
    trainer = Trainer(
        autoencoder,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        pathlib.Path(args.training.output_dir),
    )

    # Train the autoencoder
    trainer.train(
        num_epochs=args.training.epochs,
        batch_size=args.training.batch_size,
    )

    # Save the model
    trainer.save_model(args.training.epochs)
    print("Finished training the autoencoder.", flush=True)


if __name__ == "__main__":
    train_autoencoder(config)
