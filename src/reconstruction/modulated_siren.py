import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from src.encoding.custom_mri_encoder import build_autoencoder, load_model, config, CustomEncoder
import pathlib


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

    def __init__(self, latent_dim, encoder_type = 'custom'):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        if encoder_type == 'custom':
            self.encoder = CustomEncoder(pathlib.Path(r'C:\Users\jan\Documents\python_files\adlm\refactoring\models\model1.pth'))
        else:
            pass #for now

    def forward(self, x):
        return self.encoder(x)
        


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
        outer_patch_size,
        inner_patch_size,
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

        self.encoder = Encoder(latent_dim=latent_dim, encoder_type=encoder_type)

        tensors = [
            torch.linspace(-1, 1, steps=self.inner_patch_size),
            torch.linspace(-1, 1, steps=self.inner_patch_size),
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
            out, "b (h w)-> () b h w", h=self.inner_patch_size, w=self.inner_patch_size
        )
        out = out.squeeze(0)

        return out
