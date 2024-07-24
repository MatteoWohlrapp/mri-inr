import datetime
import os
import pathlib

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader


class Inspection(nn.Module):
    """Simple module to print the shape of the tensor for debugging."""

    def __init__(self, text: str = ""):
        super(Inspection, self).__init__()
        self.text = text

    def forward(self, x):
        print(self.text)
        print(x.shape, flush=True)
        return x


class EncoderBlock(nn.Module):
    """VGG inspired encoder block with batch normalization."""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(EncoderBlock, self).__init__()

        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.encoder_block(x)


class ConvBlock(nn.Module):
    """VGG inspired encoder block with batch normalization."""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(ConvBlock, self).__init()

        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.encoder_block(x)


class DecoderBlock(nn.Module):
    """VGG inspired decoder block with batch normalization using deconvolutions."""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(DecoderBlock, self).__init__()

        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.decoder_block(x)


class FullyConnectedBlock(nn.Module):
    """Fully connected block with batch normalization."""

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super(FullyConnectedBlock, self).__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the fully connected block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.fc_block(x)


class Autoencoder_v1(nn.Module):
    """Autoencoder for perceptual loss. Inout size is 24x24."""

    def __init__(self, in_channels=1, out_channels=1, img_size=24):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            img_size (int): The size of the input image.
        """
        super(Autoencoder_v1, self).__init__()

        min_kernel_size = img_size // 2 // 2 // 2

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 64),  # out 12x12
            EncoderBlock(64, 128),  # out 6x6
            EncoderBlock(128, 256),  # out 3x3
            nn.Flatten(),
            FullyConnectedBlock(256 * min_kernel_size * min_kernel_size, 512),
            FullyConnectedBlock(512, 256),
        )

        self.decoder = nn.Sequential(
            FullyConnectedBlock(256, 512),
            FullyConnectedBlock(512, 256 * min_kernel_size * min_kernel_size),
            nn.Unflatten(1, (256, 3, 3)),
            DecoderBlock(256, 128),  # out 6x6
            DecoderBlock(128, 64),  # out 12x12
            DecoderBlock(64, out_channels),  # out 24x24
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x


class Autoencoder_v2(nn.Module):
    """Simpler Autoencoder for perceptual loss. Inout size is 24x24."""

    def __init__(self, in_channels=1, out_channels=1, img_size=24):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            img_size (int): The size of the input image.
        """
        super(Autoencoder_v2, self).__init__()

        min_kernel_size = img_size // 2 // 2

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 64),  # out 12x12
            EncoderBlock(64, 128),  # out 6x6
            nn.Flatten(),
            FullyConnectedBlock(128 * min_kernel_size * min_kernel_size, 256),
        )

        self.decoder = nn.Sequential(
            FullyConnectedBlock(256, 128 * min_kernel_size * min_kernel_size),
            nn.Unflatten(1, (128, min_kernel_size, min_kernel_size)),
            DecoderBlock(128, 64),  # out 12x12
            DecoderBlock(64, out_channels),  # out 24x24
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x


class Trainer:
    """Trainer class to train the autoencoder."""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        output_dir,
    ):
        """
        Args:
            model (torch.nn.Module): The model to train.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            device (torch.device): The device to use for computation.
            train_dataset (torch.utils.data.Dataset): The training dataset.
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            output_dir (pathlib.Path): The output directory to save the model.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train(self, num_epochs, batch_size):
        """
        Train the model.

        Args:
            num_epochs (int): The number of epochs to train the model.
        """
        self.model.to(self.device)
        self.model.train()

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        for epoch in tqdm.tqdm(range(num_epochs)):
            loss = self.train_epoch()

            if (epoch + 1) % 1 == 0:
                self.save_model(epoch)
                val_loss = self.validate()
                print(
                    f"Epoch {epoch+1}/{num_epochs} Loss: {loss:.6f} Val Loss: {val_loss:.6f}"
                )

    def train_epoch(self):
        loss = 0
        with torch.cuda.amp.autocast(enabled=self.amp):
            for images, _ in self.train_loader:
                self.optimizer.zero_grad()
                images = images.to(self.device).float()

                outputs = self.model(images)
                loss = self.criterion(outputs, images)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss += loss.item()
        return loss

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
        return loss.item()

    def save_model(self, epoch):
        """
        Save the model.

        Args:
            epoch (int): The current epoch.
        """
        model_path = pathlib.Path(self.output_dir) / f"model_{epoch}.pth"
        torch.save(self.model.state_dict(), model_path)
        torch.save(
            self.optimizer.state_dict(),
            pathlib.Path(self.output_dir) / f"optimizer_{epoch}.pth",
        )


def load_autoencoder_v1(device, model_path):
    """Load the autoencoder model."""
    model = Autoencoder_v1()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def load_autoencoder_v2(device, model_path):
    """Load the autoencoder model."""
    model = Autoencoder_v2()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


class Encoder_v1(nn.Module):
    """Encoder for perceptual loss. Load the weights from the autoencoder."""

    def __init__(self, model_path, device, in_channels=1, out_channels=1, img_size=24):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            img_size (int): The size of the input image.
        """
        super(Encoder_v1, self).__init__()
        autoencoder = load_autoencoder_v1(device, model_path)
        self.encoder = autoencoder.encoder

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x


class Encoder_v2(nn.Module):
    """Encoder for perceptual loss. Load the weights from the autoencoder."""

    def __init__(self, model_path, device, in_channels=1, out_channels=1, img_size=24):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            img_size (int): The size of the input image.
        """
        super(Encoder_v2, self).__init__()
        autoencoder = load_autoencoder_v2(device, model_path)
        self.encoder = autoencoder.encoder

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x


