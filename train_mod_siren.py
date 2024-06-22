import torch
import pathlib
import types
from torch.utils.data import DataLoader
from src.data.mri_dataset import MRIDataset
from src.reconstruction.modulated_siren import ModulatedSiren
from src.reconstruction.training import Trainer
from src.util.util import time_function
import os

import torch.nn as nn
import torch.optim as optim

@time_function
def train_mod_siren(args):
    print("Training the modulated SIREN...")
    print(args)
    # Load dataset
    train_dataset = MRIDataset(pathlib.Path(args.path_train_dataset), number_of_samples = args.num_samples_train)
    val_dataset = MRIDataset(pathlib.Path(args.path_val_dataset), number_of_samples = args.num_samples_val)

    # Set the device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using the GPU")
    else:
        device = torch.device("cpu")

    # Load the model
    mod_siren = ModulatedSiren(
        dim_in=2,
        dim_hidden=256,
        dim_out=1,
        num_layers=5,
        latent_dim=256,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        dropout=0.1,
        modulate=True,
        encoder_type="custom",
        outer_patch_size=32,
        inner_patch_size=16,
    )
    mod_siren.to(device)

    # Define the criterion and optimizer
    optimizer = torch.optim.Adam(mod_siren.parameters(), lr=0.001)

    # Define the trainer
    trainer = Trainer(
        model=mod_siren,
        device=device,
        train_dataset=train_dataset,
        optimizer=optimizer,
        output_dir=args.output_dir,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        output_name=args.output_name,
        limit_io=True,
        outer_patch_size=32,
        inner_patch_size=16,
    )
    trainer.load_model(
        model_path=pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-22_08-29-55\model_checkpoints\model_epoch_100.pth"),
        optimizer_path=pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-22_08-29-55\model_checkpoints\optimizer_epoch_100.pth")
    )
    # Train the model
    trainer.train(args.epochs)

if __name__ == '__main__':
    args = {
        "path_train_dataset": r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train",
        "path_val_dataset": r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train",
        "num_samples_train": 15,
        "num_samples_val": 1,
        "device": "cuda",
        "output_dir": r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren",
        "output_name": "mod_siren_1.0",
        "batch_size": 2,
        "epochs": 80
    }
    args = types.SimpleNamespace(**args)
    train_mod_siren(args)