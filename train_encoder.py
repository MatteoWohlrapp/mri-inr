import datetime
import pathlib
import torch
import types
from src.data.mri_dataset import MRIDataset

"""
This script trains the custom encoder on the fastMRI dataset.
"""


from src.networks.encoding.custom_mri_encoder import (
    Trainer,
    build_autoencoder,
    config,
    load_model,
    save_model,
)


def train_encoder(args):
    """
    Train the encoder.

    Args:
        args (argparse.Namespace): The arguments to use for training.
    """
    print("Training the encoder...", flush=True)

    # Load dataset
    train_dataset = MRIDataset(
        pathlib.Path(args.path_train_dataset),
        number_of_samples=args.num_samples_train,
        outer_patch_size=args.outer_patch_size,
        inner_patch_size=args.inner_patch_size,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
    val_dataset = MRIDataset(
        pathlib.Path(args.path_val_dataset),
        number_of_samples=args.num_samples_val,
        outer_patch_size=args.outer_patch_size,
        inner_patch_size=args.inner_patch_size,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using the GPU")
    else:
        device = torch.device("cpu")

    # Load the model
    if args.model_path == "":
        autoencoder = build_autoencoder(config)
    else:
        print(args.model_path)
        autoencoder = load_model(pathlib.Path(args.model_path), device)

    # Define the criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)

    # Define the trainer
    trainer = Trainer(
        autoencoder,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        args.batch_size,
        args,
    )

    # Train the model
    trainer.train(args.epochs)

    # Save the model TODO change path if not given
    save_model(
        autoencoder, pathlib.Path(r"./output/custom_encoder/model1.pth"), trainer
    )


if __name__ == "__main__":
    print("start training encoder")
    args = {
        "path_train_dataset": r"../../dataset/fastmri/brain/singlecoil_train/processed_v2",
        "path_val_dataset": r"../../dataset/fastmri/brain/singlecoil_val/processed_v2",
        "num_samples_train": 0,
        "num_samples_val": 50,
        "device": "cuda",
        "model_path": r"",
        "batch_size": 400,
        "epochs": 10000,
        "output_dir": "./models",
        "output_name": "encoder_v2" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "outer_patch_size": 32,
        "inner_patch_size": 32,
    }
    args = types.SimpleNamespace(**args)
    train_encoder(args)
    print("Done training encoder")
