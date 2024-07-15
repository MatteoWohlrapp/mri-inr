"""Reimplementation of the autoencoder"""

import datetime
import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.configuration.configuration import save_config_to_yaml
from src.data.mri_sampler import MRISampler
from src.util.error import error_metrics
from src.util.tiling import image_to_patches
from src.util.util import nan_in_tensor

# Constants
tile_size = 32
latent_dim = 256

# Configuration dictionary
config = {
    "id": f"autoencoder_v2_{latent_dim}",
    "encoder": [
        {
            "type": "Conv2d",
            "params": {
                "in_channels": 1,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
            },
        },
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {
            "type": "Conv2d",
            "params": {
                "in_channels": 16,
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
            },
        },
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {
            "type": "Conv2d",
            "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 8},
        },
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {"type": "Flatten"},
        {"type": "Linear", "params": {"in_features": 64, "out_features": latent_dim}},
    ],
    "decoder": [
        {"type": "Linear", "params": {"in_features": latent_dim, "out_features": 64}},
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {"type": "Unflatten"},
        {
            "type": "ConvTranspose2d",
            "params": {"in_channels": 64, "out_channels": 32, "kernel_size": 8},
        },
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {
            "type": "ConvTranspose2d",
            "params": {
                "in_channels": 32,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
            },
        },
        {"type": "LeakyReLU", "params": {"negative_slope": 0.2}},
        {
            "type": "ConvTranspose2d",
            "params": {
                "in_channels": 16,
                "out_channels": 1,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "output_padding": 1,
            },
        },
        {"type": "Sigmoid"},
    ],
}

class Flatten(nn.Module):
    """Custom Flatten layer to transform a tensor to a 2D matrix"""
    def forward(self, input):
        return input.view(input.size(0), -1)

class Unflatten(nn.Module):
    """Custom Unflatten layer to transform a 2D matrix back to a tensor"""
    def forward(self, input):
        return input.view(-1, 64, 1, 1)

class Autoencoder(nn.Module):
    """Autoencoder model consisting of an encoder and a decoder

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        id_ (str): The ID of the autoencoder.
        latent_dim (int): The dimension of the latent space.
        config (dict): Configuration parameters for the autoencoder.

    Attributes:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        id (str): The ID of the autoencoder.
        latent_dim (int): The dimension of the latent space.
        config (dict): Configuration parameters for the autoencoder.
    """

    def __init__(self, encoder, decoder, id_, latent_dim, config):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id = id_
        self.latent_dim = latent_dim
        self.config = config

    def forward(self, x):
        """Forward pass through the autoencoder

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        nan_in_tensor(x, "input x in forward")
        batch_size = x.shape[0]
        x = x.view(-1, 1, tile_size, tile_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, -1, tile_size, tile_size)
        nan_in_tensor(x, "output x in forward")
        return x

def build_layers(layer_configs):
    """
    Helper function to build encoder and decoder layers from the configuration.

    Args:
        layer_configs (list): A list of dictionaries containing layer configurations.

    Returns:
        list: A list of PyTorch layer objects.

    Raises:
        ValueError: If an unsupported layer type is encountered.

    """
    layers = []
    for layer in layer_configs:
        layer_type = layer["type"]
        if layer_type == "Conv2d":
            layers.append(nn.Conv2d(**layer["params"]))
        elif layer_type == "ConvTranspose2d":
            layers.append(nn.ConvTranspose2d(**layer["params"]))
        elif layer_type == "LeakyReLU":
            layers.append(nn.LeakyReLU(**layer["params"]))
        elif layer_type == "Linear":
            layers.append(nn.Linear(**layer["params"]))
        elif layer_type == "Sigmoid":
            layers.append(nn.Sigmoid())
        elif layer_type == "Flatten":
            layers.append(Flatten())
        elif layer_type == "Unflatten":
            layers.append(Unflatten())
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    return layers

def build_autoencoder(config):
    """
    Builds the full autoencoder model from the configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the autoencoder.

    Returns:
        Autoencoder: The built autoencoder model.

    """
    encoder_layers = build_layers(config["encoder"])
    decoder_layers = build_layers(config["decoder"])
    return Autoencoder(
        nn.Sequential(*encoder_layers),
        nn.Sequential(*decoder_layers),
        config["id"],
    )

def save_model(autoencoder, path, trainer):
    """
    Saves the model, including the autoencoder and optimizer state.

    Args:
        autoencoder (Autoencoder): The autoencoder model to be saved.
        path (str): The path where the model will be saved.
        trainer (Trainer): The trainer object containing the optimizer state.

    Returns:
        None
    """
    torch.save(
        {
            "config": autoencoder.config,
            "state_dict": autoencoder.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        },
        path,
    )

def load_model(path, device):
    """
    Loads the model from a checkpoint.

    Args:
        path (str): The path to the checkpoint file.
        device (str): The device to load the model on.

    Returns:
        autoencoder (torch.nn.Module): The loaded autoencoder model.
    """
    checkpoint = torch.load(path, map_location=device)
    autoencoder = build_autoencoder(config)
    if "state_dict" in checkpoint:
        autoencoder.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
    return autoencoder

class Trainer:
    """Class to manage the training and validation logic"""
    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        batch_size,
        args,
    ):
        """
        Initializes a new instance of the Trainer class.

        Args:
            model (nn.Module): The neural network model.
            criterion: The loss function.
            optimizer: The optimizer.
            device: The device to run the training on.
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            batch_size: The batch size for training.
            args: Additional arguments.

        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.name = (
            f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{self.model.id}'
        )
        self.dir = pathlib.Path(f"./output/custom_encoder/{self.name}")
        self.dir.mkdir()
        save_config_to_yaml(args, self.dir / "config.yaml")
        self.writer = SummaryWriter(log_dir=f"runs/tensorboard/{self.name}")

    def process_batch(self, batch_fullysampled, batch_undersampled):
            """
            Processes a single batch through the model and computes the loss.

            Args:
                batch_fullysampled (torch.Tensor): The fully sampled batch.
                batch_undersampled (torch.Tensor): The undersampled batch.

            Returns:
                torch.Tensor: The output of the model.
                torch.Tensor: The computed loss.
            """
            batch_undersampled = batch_undersampled.to(self.device).float()
            batch_fullysampled = batch_fullysampled.to(self.device).float()
            self.optimizer.zero_grad()
            output = self.model(batch_undersampled).squeeze(1)
            loss = self.criterion(output, batch_fullysampled)
            return output, loss

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            epoch (int): The current epoch number.

        Returns:
            None
        """
        self.model.train()
        for i, (batch_fullysampled, batch_undersampled) in enumerate(train_loader):
            output, loss = self.process_batch(batch_fullysampled, batch_undersampled)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar(
                "training_loss", loss.item(), epoch * len(train_loader) + i
            )

    def validate_one_epoch(self, val_loader, epoch):
        """Validates the model for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): The data loader for validation dataset.
            epoch (int): The current epoch number.

        Returns:
            float: The validation loss for the epoch.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (batch_fullysampled, batch_undersampled) in enumerate(val_loader):
                output, loss = self.process_batch(
                    batch_fullysampled, batch_undersampled
                )
                val_loss += loss.item()
        self.writer.add_scalar("val_loss", val_loss, epoch)
        return val_loss

    def train(self, num_epochs):
            """
            Trains the model for a specified number of epochs.

            Args:
                num_epochs (int): The number of epochs to train the model.

            Returns:
                None
            """
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            for epoch in range(num_epochs):
                self.train_one_epoch(train_loader, epoch)
                val_loss = self.validate_one_epoch(val_loader, epoch)
                print(f"Epoch {epoch}, Val Loss: {val_loss:.5f}", flush=True)
                if epoch % 10 == 0:
                    save_model(self.model, f"{str(self.dir)}/epoch_{epoch}.pth", self)
            save_model(self.model, f"{str(self.dir)}/epoch_{num_epochs}.pth", self)
            self.writer.close()

class CustomEncoder(nn.Module):
    """Encoder model extracted from the autoencoder model

    Args:
        autoencoder_path (str): Path to the pre-trained autoencoder model
        device (str): Device to load the model on (e.g., 'cpu', 'cuda')

    Attributes:
        encoder (nn.Module): Encoder module of the autoencoder
        latent_dim (int): Dimension of the latent space

    """

    def __init__(self, autoencoder_path, device):
        super(CustomEncoder, self).__init__()
        autoencoder = load_model(autoencoder_path, device)
        self.encoder = autoencoder.encoder
        self.latent_dim = autoencoder.latent_dim

    def forward(self, x):
        """Forward pass through the encoder

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation of the input tensor

        """
        x = x.view(-1, 1, tile_size, tile_size)
        x = self.encoder(x)
        return x

def test_autoencoder(test_config):
    """
    Function to test the autoencoder.

    Args:
        test_config (object): Configuration object for testing.

    Returns:
        None
    """
    print("Testing the Autoencoder...")
    print(test_config)
    output_dir = f"{test_config.testing.output_dir}/{test_config.testing.output_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Set the device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dataset
    sampler = MRISampler(pathlib.Path(test_config.data.dataset), test_config.data.test_files)

    # Load the model
    model = load_model(pathlib.Path(test_config.testing.model_path), device)

    with torch.no_grad():
        model.eval()

        for i in range(test_config.data.num_samples):
            print(f"Processing sample {i + 1}/{test_config.data.num_samples}...")
            # Load the image
            fully_sampled_img, undersampled_img, filename = sampler.get_random_sample()

            # Unsqueeze image to add batch dimension
            fully_sampled_img = fully_sampled_img.unsqueeze(0).float().to(device)
            undersampled_img = undersampled_img.unsqueeze(0).float().to(device)

            fully_sampled_patch, _ = image_to_patches(
                fully_sampled_img,
                test_config.model.outer_patch_size,
                test_config.model.inner_patch_size,
            )
            undersampled_patch, undersampled_information = image_to_patches(
                undersampled_img,
                test_config.model.outer_patch_size,
                test_config.model.inner_patch_size,
            )

            output_dir_temp = os.path.join(output_dir, filename)
            if not os.path.exists(output_dir_temp):
                os.makedirs(output_dir_temp)

            error_metrics(
                model,
                output_dir_temp,
                filename,
                fully_sampled_patch,
                undersampled_patch,
                undersampled_information,
                device,
                test_config.model.outer_patch_size,
                test_config.model.inner_patch_size,
                test_config.model.siren_patch_size,
            )
