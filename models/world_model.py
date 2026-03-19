from numpy import int32
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64

        self.conv1 = nn.Conv2d(4, 48, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2, padding=1)

        self.flatten = torch.nn.Flatten()
        

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)         # (1, C_enc, H_enc, W_enc)
            self.conv_output_shape = feats.shape[1:]   # (C_enc, H_enc, W_enc)
            self.flattened_dim = feats.numel() // 1    # C_enc * H_enc * W_enc
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        self.fc_enc = nn.Linear(self.flattened_dim, embed_dim)  # ADD THIS

        self.deconv1 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(48, observation_shape[0], kernel_size=4, stride=2, padding=1)

        self.fc_dec = nn.Linear(embed_dim, self.flattened_dim) 

        self.reward_pred = nn.Linear(embed_dim, 1)
        self.action_pred = nn.Linear(embed_dim, 3)


        # self.conv3 = nn.Conv2d()

        print(f"VAE network initialized. Input shape: {observation_shape}")


    def get_output_shape(self):
        return self.conv_output_shape


    def _conv_features(self, x):
        # Convert uint8 to float if needed (for initialization)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        return x  # (B, C_enc, H_enc, W_enc)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x

    def _deconv_forward(self, x):
        x = x.view(-1, *self.conv_output_shape)
        x = F.elu(self.deconv1(x))
        x = F.elu(self.deconv2(x))
        x = F.elu(self.deconv3(x))
        x = F.elu(self.deconv4(x))
       
        return x
        
    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        x = self._conv_forward(x)
        x = self.fc_enc(x)

        reward_pred = self.reward_pred(x)

        next_frame_pred = self.fc_dec(x)
        next_frame_pred = self._deconv_forward(next_frame_pred)
        next_frame_pred = torch.sigmoid(next_frame_pred)
        
        return next_frame_pred, reward_pred 
    



