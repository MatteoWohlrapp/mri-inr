import torch
import torch.nn as nn
from src.networks.encoding.custom_mri_encoder import CustomEncoder


# perceptual loss function using a pretrained custom mri encoder
class PerceptualLoss(nn.Module):
    def __init__(self, encoder_path, criterion, device):
        super(PerceptualLoss, self).__init__()
        self.encoder = self._load_encoder(encoder_path, device)
        self.criterion = criterion

    def _load_encoder(self, encoder_path, device):
        encoder = CustomEncoder(encoder_path, device)
        encoder.to(device)
        
        # Disable gradients for the encoder
        for param in encoder.parameters():
            param.requires_grad = False
        
        return encoder

    def forward(self, x, y):
        print(x.shape, y.shape, flush=True)
        x_encoded = self.encoder(x)
        y_encoded = self.encoder(y)
        return self.criterion(x_encoded, y_encoded)

