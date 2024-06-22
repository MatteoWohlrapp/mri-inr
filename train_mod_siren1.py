'''Change the args loading to load them from a yaml file'''
import torch
import pathlib
import types
from torch.utils.data import DataLoader
from src.data.mri_dataset import MRIDataset
from src.reconstruction.modulated_siren import ModulatedSiren
from src.reconstruction.training import Trainer
from src.util.util import time_function
import os
import yaml
from src.configuration.simple_conf import load_configuration
import torch.nn as nn
import torch.optim as optim

@time_function
def train_mod_siren(args):
    print("Training the modulated SIREN...")
    print(args)
    # Load dataset
    train_dataset = MRIDataset(pathlib.Path(args.data.train.dataset), number_of_samples = args.data.train.num_samples)
    val_dataset = MRIDataset(pathlib.Path(args.data.val.dataset), number_of_samples = args.data.val.num_samples)

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using the GPU")
    else:
        device = torch.device("cpu")

    # Load the model
    mod_siren = ModulatedSiren(
        dim_in=args.model.dim_in,   
        dim_hidden=args.model.dim_hidden,
        dim_out=args.model.dim_out,
        num_layers=args.model.num_layers,
        latent_dim=args.model.latent_dim,
        w0=args.model.w0,
        w0_initial=args.model.w0_initial,
        use_bias=args.model.use_bias,
        dropout=args.model.dropout,
        modulate=True,
        encoder_type=args.model.encoder_type,
        outer_patch_size=args.model.outer_patch_size,
        inner_patch_size=args.model.inner_patch_size,
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
        output_dir=args.training.output_dir,
        val_dataset=val_dataset,
        batch_size=args.training.batch_size,
        output_name=args.training.output_name,
        limit_io=True,
        outer_patch_size=args.model.outer_patch_size,
        inner_patch_size=args.model.inner_patch_size,
    )
    trainer.load_model(
        model_path=pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-22_08-29-55\model_checkpoints\model_epoch_100.pth"),
        optimizer_path=pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\refactoring\output\mod_siren\mod_siren_2024-06-22_08-29-55\model_checkpoints\optimizer_epoch_100.pth")
    )
    # Train the model
    trainer.train(args.training.epochs)

if __name__ == '__main__':
    args = load_configuration(pathlib.Path("./src/configuration/train_modulated_siren.yaml"))
    print(args.data.train.dataset)
    train_mod_siren(args)