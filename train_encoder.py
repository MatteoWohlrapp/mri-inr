import torch
from src.data.mri_dataset import MRIDataset
from src.encoding.custom_mri_encoder import Trainer, build_autoencoder, config, load_model, save_model
from src.util.util import time_function
import pathlib
import torch.nn as nn
import types

@time_function
def train_encoder(args):
    print("Training the encoder...")
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
    if args.model_path == "":
        autoencoder = build_autoencoder(config)
    else:
        autoencoder = load_model(pathlib.Path(args.model_path))

    # Define the criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Define the trainer
    trainer = Trainer(
        autoencoder,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        args.batch_size,
    )

    # Train the model
    trainer.train(args.epochs)

    # Save the model TODO change path if not given
    save_model(autoencoder, pathlib.Path(args.model_path))

if __name__ == '__main__':
    args = {
        "path_train_dataset": r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train",
        "path_val_dataset": r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_train",
        "num_samples_train": 15,
        "num_samples_val": 2,
        "device": "cuda",
        "model_path": r"C:\Users\jan\Documents\python_files\adlm\refactoring\models\model1.pth",
        "batch_size": 10,
        "epochs": 1000
    }
    args = types.SimpleNamespace(**args)
    train_encoder(args)