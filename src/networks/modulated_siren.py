"""
Modulated Siren network
"""

import math
import pathlib

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.networks.encoding.siren_encoder import CustomEncoder, FixedEncoder
from src.networks.encoding.vgg import VGGAutoEncoder, get_configs, load_dict


def cast_tuple(val, repeat=1):
    """
    Cast a value to a tuple.

    Args:
        val (Any): The value to cast.
        repeat (int): The number of times to repeat the value.

    Returns:
        Tuple: The value as a tuple
    """
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    """Sine activation function."""

    def __init__(self, w0=1.0):
        """
        Initialize the Sine activation function.

        Args:
            w0 (float): The frequency of the sine function.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """
        Forward pass of the Sine activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.sin(self.w0 * x)


class Morlet(nn.Module):
    """Morlet activation function."""

    def __init__(self, w0=1.0):
        """
        Initialize the Morlet activation function.

        Args:
            w0 (float): The frequency of the Morlet function.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """
        Forward pass of the Morlet activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.sin(self.w0 * x) * torch.exp(-0.5 * x**2)


class Siren(nn.Module):
    """SIREN layer."""

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        """
        Initialize a SIREN layer.

        Args:
            dim_in (int): The input dimension.
            dim_out (int): The output dimension.
            w0 (float): The frequency of the sine function.
            c (float): The constant for initializing the weights.
            is_first (bool): Whether this is the first layer.
            use_bias (bool): Whether to use a bias term.
            activation (nn.Module): The activation function to use.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        if activation == "morlet":
            self.activation = Morlet(w0)
        else:
            self.activation = Sine(w0)
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        """
        Initialize the weights and bias.

        Args:
            weight (torch.Tensor): The weight tensor.
            bias (torch.Tensor): The bias tensor.
            c (float): The constant for initializing the weights.
            w0 (float): The frequency of the sine function.
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        """
        Forward pass of the SINE layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class SirenNet(nn.Module):
    """SIREN network."""

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0,
        w0_initial,
        use_bias,
        dropout,
        activation,
    ):
        """
        Initialize a SIREN network.

        Args:
            dim_in (int): The input dimension.
            dim_hidden (int): The hidden dimension.
            dim_out (int): The output dimension.
            num_layers (int): The number of layers.
            w0 (float): The frequency of the sine function.
            w0_initial (float): The frequency of the sine function for the first layer.
            use_bias (bool): Whether to use a bias term.
            dropout (float): The dropout rate.
            activation (str): The activation function to use.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout,
                activation=activation,
            )

            self.layers.append(layer)

        self.last_layer = Siren(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias
        )

    def forward(self, x, mods=None):
        """
        Forward pass of the SIREN network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if mod is not None:
                x *= rearrange(mod, "b d -> b () d")

        return self.last_layer(x)


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self, latent_dim, encoder_path, device, encoder_type="custom"):
        """
        Initialize an encoder network.

        Args:
            latent_dim (int): The latent dimension.
            encoder_path (str): The path to the encoder.
            device (torch.device): The device to use.
            encoder_type (str): The type of encoder.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        if encoder_type == "custom":
            self.encoder = FixedEncoder(pathlib.Path(encoder_path), device)
            self.encoder.train()
            self.fc = nn.Identity()
        elif encoder_type == "vgg":
            self.encoder, num_features = self.load_vgg(encoder_path)
            self.encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.fc = nn.Linear(num_features, latent_dim)

    def load_vgg(self, encoder_path):
        """
        Load a vgg encoder

        Args:
            encoder_path (str): The path to the encoder.

        Returns:
            nn.Module: The encoder.
            int: The number of features
        """
        model = VGGAutoEncoder(get_configs("vgg16"))
        load_dict(encoder_path, model)

        num_features = 512 * 7 * 7

        return model.encoder, num_features

    def forward(self, x):
        """
        Forward pass of the econder.

        Args:
            x (torch.Tensor): The input image patches.

        Returns:
            torch.Tensor: The latent code.
        """
        if self.encoder_type == "custom":
            x = self.encoder(x)
        elif self.encoder_type == "vgg":
            x = x.unsqueeze(1)
            x = self.encoder(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


class Modulator(nn.Module):
    """Modulator network."""

    def __init__(self, dim_in, dim_hidden, num_layers):
        """
        Initialize a modulator network.

        Args:
            dim_in (int): The input dimension.
            dim_hidden (int): The hidden dimension.
            num_layers (int): The number of layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z):
        """
        Forward pass of the Modulator network.

        Args:
            z (torch.Tensor): The input latent code.

        Returns:
            torch.Tensor: The modulations.
        """
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)


class ModulatedSiren(nn.Module):
    """Modulated SIREN network."""

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        latent_dim,
        w0,
        w0_initial,
        use_bias,
        dropout,
        modulate,
        encoder_type,
        encoder_path,
        outer_patch_size,
        inner_patch_size,
        siren_patch_size,
        device,
        activation,
    ):
        """
        Initialize a modulated SIREN network.

        Args:
            dim_in (int): The input dimension.
            dim_hidden (int): The hidden dimension.
            dim_out (int): The output dimension.
            num_layers (int): The number of layers.
            latent_dim (int): The latent dimension.
            w0 (float): The frequency of the sine function.
            w0_initial (float): The frequency of the sine function for the first layer.
            use_bias (bool): Whether to use a bias term.
            dropout (float): The dropout rate.
            modulate (bool): Whether to modulate the network.
            encoder_type (str): The type of encoder.
            encoder_path (str): The path to the encoder.
            outer_patch_size (int): The size of the outer patch.
            inner_patch_size (int): The size of the inner patch.
            siren_patch_size (int): The size of the SIREN patch.
            device (torch.device): The device to use.
            activation (str): The activation function to use.
        """
        super().__init__()

        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.modulate = modulate
        self.encoder_type = encoder_type
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.siren_patch_size = siren_patch_size
        self.activation = activation

        self.net = SirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            use_bias=use_bias,
            dropout=dropout,
            activation=activation,
        )

        self.modulator = Modulator(
            dim_in=latent_dim, dim_hidden=dim_hidden, num_layers=num_layers
        )

        self.encoder = Encoder(
            latent_dim=latent_dim,
            encoder_path=encoder_path,
            device=device,
            encoder_type=encoder_type,
        )

        tensors = [
            torch.linspace(-1, 1, steps=self.siren_patch_size),
            torch.linspace(-1, 1, steps=self.siren_patch_size),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w b -> (h w) b")
        self.register_buffer("grid", mgrid)

    def forward(self, tiles):
        """
        Forward pass of the modulated SIREN network.

        Args:
            tiles (torch.Tensor): The input image patches.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size = tiles.shape[0]
        mods = self.modulator(self.encoder(tiles))

        coords = self.grid.clone().detach().repeat(batch_size, 1, 1).requires_grad_()

        out = self.net(coords, mods)
        out = out.squeeze(2)
        out = rearrange(
            out, "b (h w)-> () b h w", h=self.siren_patch_size, w=self.siren_patch_size
        )
        out = out.squeeze(0)

        return out
