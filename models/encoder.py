from numpy import int32
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class Encoder(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024):
        super().__init__()

        # Encoder conv layers - track channel dimensions for decoder
        # NO spatial compression - all stride-1 layers: 96x96 -> 96x96
        # Ball preserved at FULL 2-3 pixel resolution (no degradation!)
        # 16 channels × 96×96 = 147,456 dims → ~250M params total, fits in 12GB GPU
        self.conv_channels = [3, 8, 16]

        self.conv1 = nn.Conv2d(self.conv_channels[0], self.conv_channels[1], kernel_size=3, stride=2, padding=1)  # 96->48
        self.conv2 = nn.Conv2d(self.conv_channels[1], self.conv_channels[2], kernel_size=3, stride=2, padding=1)  # 48->24

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)         # (1, C_enc, H_enc, W_enc)
            self.conv_output_shape = feats.shape[1:]   # (C_enc, H_enc, W_enc)
            self.flattened_dim = feats.numel() // 1    # C_enc * H_enc * W_enc
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        self.fc_enc = nn.Linear(self.flattened_dim, embed_dim)
        
        print(f"VAE network initialized. Input shape: {observation_shape}")

    def get_output_shape(self):
        return self.conv_output_shape

    def get_conv_channels(self):
        return self.conv_channels

    def _conv_features(self, x):
        # Convert uint8 to float if needed (for initialization)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        return x  # (B, 16, 24, 24)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x
        
    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        x = self._conv_forward(x)
        x = self.fc_enc(x)
        return x
    


class Decoder(BaseModel):

    def __init__(self, observation_shape, embed_dim, conv_output_shape=(16, 24, 24), conv_channels=[3, 8, 16]):
        super().__init__()

        # Use the encoder's conv output shape
        self.conv_output_shape = conv_output_shape
        conv_flat_size = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]

        self.fc_dec = nn.Linear(embed_dim, conv_flat_size)

        # Build decoder layers dynamically in reverse order from encoder
        # Use output_padding=1 to recover 96x96 from 24x24
        self.deconv1 = nn.ConvTranspose2d(conv_channels[2], conv_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)  # 24->48
        self.deconv2 = nn.ConvTranspose2d(conv_channels[1], conv_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)  # 48->96

    
    def _deconv_forward(self, x):
        x = x.view(-1, *self.conv_output_shape)
        x = F.elu(self.deconv1(x))
        # No activation on final layer - let sigmoid handle it
        x = self.deconv2(x)

        return x

    def forward(self, x):
        x = self.fc_dec(x)
        x = self._deconv_forward(x)
        x = torch.sigmoid(x)
        return x


