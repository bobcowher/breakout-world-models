from numpy import int32, rec
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.base import BaseModel
from models.ssim_loss import ssim_loss
from models.dynamics_model import DynamicsModel

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, action_dim=128, n_actions=4, feature_dim=1024):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64
        # Encoder/Decoder for observations
        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(observation_shape=observation_shape, embed_dim=feature_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        # Dynamics model - predicts next embedding from current embedding + action
        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=512)

        # Prediction heads
        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)  # embed + action → reward
        self.action_pred = nn.Linear(embed_dim, n_actions)      # embed → action logits
        self.done_pred = nn.Linear(embed_dim + n_actions, 1)    # embed + action → done probability

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        print(f"World Model initialized. Input shape: {observation_shape}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Dynamics: embed + action → next_embed")
        print(f"  Prediction heads: reward, action, done")


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

    def compute_loss(self, obs, actions, rewards, next_obs, dones):
        """
        Compute all world model losses.

        Args:
            obs: (B, C, H, W) uint8 observations
            actions: (B,) action indices
            rewards: (B,) rewards
            next_obs: (B, C, H, W) uint8 next observations
            dones: (B,) done flags

        Returns:
            combined_loss: scalar total loss
            loss_dict: dictionary of individual losses
        """
        # Normalize observations
        obs_normalized = obs.float() / 255.0
        next_obs_normalized = next_obs.float() / 255.0

        if obs_normalized.ndim == 5:
            obs_normalized = obs_normalized.squeeze(1)
        if next_obs_normalized.ndim == 5:
            next_obs_normalized = next_obs_normalized.squeeze(1)

        # Convert actions to one-hot
        batch_size = obs.shape[0]
        action_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()

        # Forward pass
        recon, embeds, next_embed_pred, reward_pred, action_pred, done_pred = self.forward(obs_normalized, action_onehot)

        # === 1. Reconstruction Loss ===
        l1_loss = F.l1_loss(recon, obs_normalized)
        structural_loss = ssim_loss(recon, obs_normalized)
        recon_loss = l1_loss + 0.2 * structural_loss

        # === 2. Dynamics Loss ===
        # Encode next observation to get target embedding
        next_embeds = self.encode(next_obs_normalized)  # (B, 1, embed_dim)
        next_embed_target = next_embeds.view(-1, next_embeds.shape[-1])  # (B, embed_dim)

        # MSE between predicted and actual next embedding
        dynamics_loss = F.mse_loss(next_embed_pred, next_embed_target.detach())

        # === 3. Reward Loss ===
        # Reward is in range [-1, 0, 1] after life penalty
        reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards.float())

        # === 4. Action Loss ===
        # Predict what action was actually taken (inverse model)
        action_loss = F.cross_entropy(action_pred, actions.long())

        # === 5. Done Loss ===
        # Binary classification
        done_loss = F.binary_cross_entropy(done_pred.squeeze(-1), dones.float())

        # === Combined Loss ===
        combined_loss = (
            1.0 * recon_loss +
            1.0 * dynamics_loss +
            1.0 * reward_loss +
            0.5 * action_loss +
            0.5 * done_loss
        )

        return combined_loss, {
            "total": combined_loss.item(),
            "recon": recon_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward": reward_loss.item(),
            "action": action_loss.item(),
            "done": done_loss.item(),
            "l1": l1_loss.item(),
            "ssim": structural_loss.item(),
        }


    def forward(self, obs, action_onehot):
        """
        Full forward pass through world model.

        Args:
            obs: (B, C, H, W) observations (uint8 or normalized float)
            action_onehot: (B, n_actions) one-hot encoded actions

        Returns:
            recon: (B, C, H, W) reconstructed observation
            embeds: (B, 1, embed_dim) current state embeddings
            next_embed_pred: (B, embed_dim) predicted next state embedding
            reward_pred: (B, 1) predicted reward
            action_pred: (B, n_actions) predicted action logits
            done_pred: (B, 1) predicted done probability
        """
        # Encode observation to latent state
        embeds = self.encode(obs)  # (B, 1, embed_dim)

        # Decode for reconstruction
        embeds_flat = embeds.view(-1, embeds.shape[-1])  # (B, embed_dim)
        recon = self.decode(embeds_flat)  # (B, C, H, W)

        # Flatten embeddings for predictions
        embed = embeds_flat  # (B, embed_dim)

        # Predict next embedding using dynamics model
        next_embed_pred = self.dynamics(embed, action_onehot)  # (B, embed_dim)

        # Predict action from current state (policy)
        action_pred = self.action_pred(embed)  # (B, n_actions)

        # Predict reward from current state + action
        embed_action = torch.cat([embed, action_onehot], dim=-1)
        reward_pred = torch.tanh(self.reward_pred(embed_action))  # (B, 1) in [-1, 1]

        # Predict done from current state + action
        done_pred = torch.sigmoid(self.done_pred(embed_action))  # (B, 1) in [0, 1]

        return recon, embeds, next_embed_pred, reward_pred, action_pred, done_pred
    



