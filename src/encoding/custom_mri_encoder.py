"""Reimplementation of the autoencoder """

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
from src.util.tiling import collate_fn

tile_size = 32
latent_dim = 256

config = {
    "id": f"autoencoder_v1_{latent_dim}",
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
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 64, 1, 1)


class Inspector(nn.Module):
    def forward(self, input):
        print(f"Inspector input shape: {input.shape}")
        return input


class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder, id_):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id = id_
        self.latent_dim = latent_dim
        self.config = config

    def forward(self, x):
        batch_size = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        x = x.view(-1, 1, tile_size, tile_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, -1, tile_size, tile_size)
        return x

def build_autoencoder(config):
    def build_layers(layer_configs):
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
            elif layer_type == "Inspector":
                layers.append(Inspector())
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return layers

    encoder_layers = build_layers(config["encoder"])
    decoder_layers = build_layers(config["decoder"])
    return Autoencoder(
        nn.Sequential(*encoder_layers),
        nn.Sequential(*decoder_layers),
        config["id"],
    )

def save_model(autoencoder, path):
    print(path)
    torch.save({
        'config': autoencoder.config,
        'state_dict': autoencoder.state_dict()
    }, path)

def load_model(path):
    checkpoint = torch.load(path)
    config = checkpoint['config']
    autoencoder = build_autoencoder(config)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    return autoencoder

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        batch_size,
    ):
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
        self.writer = SummaryWriter(log_dir=f"runs/tensorboard/{self.name}")

    def process_batch(self, batch_fullysampled, batch_undersampled):
        batch_undersampled = batch_undersampled.to(self.device).float()
        batch_fullysampled = batch_fullysampled.to(self.device).float()
        self.optimizer.zero_grad()
        output = self.model(batch_undersampled).squeeze(1)
        loss = self.criterion(output, batch_fullysampled)
        return output, loss

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        for i, (batch_fullysampled, batch_undersampled) in enumerate(train_loader):
            output, loss = self.process_batch(batch_fullysampled, batch_undersampled)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar(
                "training_loss", loss.item(), epoch * len(train_loader) + i
            )

    def validate_one_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (batch_fullysampled, batch_undersampled) in enumerate(val_loader):
                output, loss = self.process_batch(batch_fullysampled, batch_undersampled)
                val_loss += loss.item()
        self.writer.add_scalar("val_loss", val_loss, epoch)
        return val_loss

    def train(self, num_epochs):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
        )
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader, epoch)
            val_loss = self.validate_one_epoch(val_loader, epoch)
            print(f"Epoch {epoch}, Val Loss: {val_loss:.5f}")
        self.writer.close()

class CustomEncoder(nn.Module):
    """Encoder model extracted from the autoencoder model"""
    def __init__(self, autoencoder_path):
        super(CustomEncoder, self).__init__()
        autoencoder = load_model(autoencoder_path)
        self.encoder = autoencoder.encoder
        self.latent_dim = autoencoder.latent_dim

    def forward(self, x):
        x = x.view(-1, 1, tile_size, tile_size)
        x = self.encoder(x)
        return x

