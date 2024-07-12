"""
Loss functions for training the MRI reconstruction model.
"""

import torch.nn as nn
from src.networks.encoding.custom_mri_encoder import CustomEncoder


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
        encoder = CustomEncoder(encoder_path, device)
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
        print(x.shape, y.shape, flush=True)
        x_encoded = self.encoder(x)
        y_encoded = self.encoder(y)
        return self.criterion(x_encoded, y_encoded)
