"""
Loss functions for training the MRI reconstruction model.
"""

import torch.nn as nn
import torch
from src.networks.encoding.custom_mri_encoder import CustomEncoder
from src.networks.encoding.new_encoder import Encoder_v2
import torch.nn.functional as F


# perceptual loss function using a pretrained custom mri encoder
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using a pretrained custom MRI encoder.
    """

    def __init__(self, encoder_path, criterion, device):
        super(PerceptualLoss, self).__init__()
        self.encoder = self._load_encoder(encoder_path, device)
        self.criterion = criterion

    def _load_encoder(self, encoder_path, device):
        """
        Load the pretrained encoder.

        Args:
            encoder_path (str): The path to the encoder.
            device (torch.device): The device to use for training.

        Returns:
            nn.Module: The pretrained encoder.
        """
        encoder = Encoder_v2(encoder_path, device)
        encoder.to(device)

        # Disable gradients for the encoder
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder

    def forward(self, x, y):
        """
        Forward pass of the perceptual loss.

        Args:
            x (torch.Tensor): The input image.
            y (torch.Tensor): The target image.

        Returns:
            torch.Tensor: The perceptual loss.
        """
        x_encoded = self.encoder(x)
        y_encoded = self.encoder(y)
        return self.criterion(x_encoded, y_encoded)


class EdgeLoss(nn.Module):
    """
    Edge loss using a Sobel filter.
    """

    def __init__(self, criterion, device):
        super(EdgeLoss, self).__init__()
        self.device = device
        self.criterion = criterion
        self.sobel_kernel_x = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_kernel_y = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x, y):
        """
        Forward pass of the edge loss.

        Args:
            x (torch.Tensor): The input image.
            y (torch.Tensor): The target image.

        Returns:
            torch.Tensor: The perceptual loss.
        """
        batch_size = x.shape[0]
        sobel_output_x_x = F.conv2d(
            x,
            self.sobel_kernel_x.to(self.device).repeat(batch_size, 1, 1, 1),
            padding=1,
            groups=batch_size,
        )
        sobel_output_y_x = F.conv2d(
            x,
            self.sobel_kernel_y.to(self.device).repeat(batch_size, 1, 1, 1),
            padding=1,
            groups=batch_size,
        )

        sobel_output_x_y = F.conv2d(
            y,
            self.sobel_kernel_x.to(self.device).repeat(batch_size, 1, 1, 1),
            padding=1,
            groups=batch_size,
        )
        sobel_output_y_y = F.conv2d(
            y,
            self.sobel_kernel_y.to(self.device).repeat(batch_size, 1, 1, 1),
            padding=1,
            groups=batch_size,
        )

        return 0.5 * self.criterion(x, y) + 0.5 * (
            self.criterion(sobel_output_x_x, sobel_output_x_y)
            + self.criterion(sobel_output_y_x, sobel_output_y_y)
        )
