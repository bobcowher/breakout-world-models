from numpy import int32, rec
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.base import BaseModel
from models.perceptual_loss import PerceptualLoss

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, action_dim=128, n_actions=4, feature_dim=1024):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64
        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(observation_shape=observation_shape, embed_dim=feature_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        self.action_input = nn.Linear(n_actions, action_dim)

        self.flatten = torch.nn.Flatten()
        
        self.reward_pred = nn.Linear(embed_dim + action_dim, 1)
        self.action_pred = nn.Linear(embed_dim, n_actions)
        self.done_pred = nn.Linear(embed_dim + action_dim, 1)

        # Perceptual loss for better reconstruction quality
        self.perceptual_loss = PerceptualLoss()

        print(f"VAE network initialized. Input shape: {observation_shape}")


    def encode(self, obs):
        # If obs is [B, C, H, W], add sequence dimension -> [B, 1, C, H, W]
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        batch_size, sequence_length = obs.shape[:2]
        obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:])
        embed_flat = self.encoder(obs_flat)
        embeds = embed_flat.view(batch_size, sequence_length, -1)

        return embeds

    def decode(self, embeds):
        return self.decoder(embeds)
        return self.conv_output_shape

    def compute_loss(self, obs, actions, rewards, continues):

        obs_normalized = obs.float() / 255.0

        recon, embeds, reward_pred, action_pred, done_pred = self.forward(obs, actions)

        # Recon is [B, C, H, W], obs_normalized might be [B, C, H, W] or [B, 1, C, H, W]
        if obs_normalized.ndim == 5:
            obs_normalized = obs_normalized.squeeze(1)

        # Combine MSE and perceptual loss for sharp, detailed reconstructions
        mse_loss = F.mse_loss(recon, obs_normalized)
        perceptual = self.perceptual_loss(recon, obs_normalized)

        # Weight perceptual loss lower since it's typically larger magnitude
        recon_loss = mse_loss + 0.01 * perceptual

        combined_loss = recon_loss

        return combined_loss, {
            "recon": recon_loss.item(),
            "mse": mse_loss.item(),
            "perceptual": perceptual.item(),
        }


    def forward(self, obs, action):
        # x: (B,3,H,W) in [0,1]
        embeds = self.encode(obs)

        # Decode from embeddings (squeeze sequence dim for decoder)
        embeds_flat = embeds.view(-1, embeds.shape[-1])
        recon = self.decode(embeds_flat)

        # Predict action from observation only (before concatenating with action)
        # action_pred = self.action_pred(x)
        #
        # y = self.action_input(action)
        # x = torch.cat([x, y], dim=1)
        #
        # reward_pred = torch.tanh(self.reward_pred(x))  # Output in range [-1, 1]
        # done_pred = self.done_pred(x)
        #
        # next_frame_pred = self.fc_dec(x)
        # next_frame_pred = self._deconv_forward(next_frame_pred)
        # next_frame_pred = torch.sigmoid(next_frame_pred)
        #
        reward_pred, action_pred, done_pred = 0, 0, 0

        return recon, embeds, reward_pred, action_pred, done_pred
    



