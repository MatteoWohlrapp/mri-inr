import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from src.networks.encoding.custom_mri_encoder import (
    CustomEncoder,
)
import pathlib
from src.networks.encoding.vgg import VGGAutoEncoder, get_configs, load_dict


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# one siren layer
class Siren(nn.Module):
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
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, dim_out, num_layers, w0, w0_initial, use_bias, dropout
    ):
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
            )

            self.layers.append(layer)

        self.last_layer = Siren(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if mod is not None:
                x *= rearrange(mod, "b d -> b () d")

        return self.last_layer(x)


# encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim, encoder_path, device, encoder_type="custom"):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        if encoder_type == "custom":
            self.encoder = CustomEncoder(pathlib.Path(encoder_path), device)
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
        model = VGGAutoEncoder(get_configs("vgg16"))
        load_dict(encoder_path, model)

        num_features = 512 * 7 * 7

        return model.encoder, num_features

    def forward(self, x):

        if self.encoder_type == "custom":
            x = self.encoder(x)
        elif self.encoder_type == "vgg":
            print(x.shape)
            x = x.unsqueeze(1)
            print(x.shape)
            x = self.encoder(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            print(x.shape)

        x = self.fc(x)
        return x


# modulatory feed forward network
class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)


# complete network
class ModulatedSiren(nn.Module):
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
    ):
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

        self.net = SirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            use_bias=use_bias,
            dropout=dropout,
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
