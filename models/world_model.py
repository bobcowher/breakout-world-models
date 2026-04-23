from numpy import int32, rec
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.base import BaseModel
from models.ssim_loss import ssim_loss
from models.dynamics_model import DynamicsModel


def gradient_loss(pred, target):
    """
    Compute gradient/edge loss between predicted and target images.

    Preserves high-contrast edges and small objects (like the ball).
    Computes L1 loss on horizontal and vertical gradients.

    Args:
        pred: (B, C, H, W) predicted images
        target: (B, C, H, W) target images

    Returns:
        scalar gradient loss
    """
    # Horizontal gradients (edges)
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Vertical gradients (edges)
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    # L1 loss on gradients
    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)

    return loss_dx + loss_dy

class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, action_dim=128, n_actions=4, feature_dim=None,
                 embed_norm='layernorm'):
        """
        Args:
            embed_norm: Normalization strategy for embeddings. Options:
                - 'layernorm': Layer normalization (default, most stable)
                - 'tanh': Tanh activation (bounds to [-1, 1])
                - 'l2': L2 normalization (projects to unit sphere)
                - None: No normalization (original behavior)
        """
        super().__init__()

        if feature_dim is None:
            feature_dim = embed_dim

        # print(observation_shape[-1])
        # conv_output_dim = 64
        # Encoder/Decoder for observations
        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)
        self.decoder = Decoder(observation_shape=observation_shape, embed_dim=feature_dim,
                               conv_output_shape=self.encoder.get_output_shape(),
                               conv_channels=self.encoder.get_conv_channels())

        # Dynamics model - predicts next embedding from current embedding + action
        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=2048)

        # Embedding normalization (applied after encoder and dynamics)
        self.embed_norm_type = embed_norm
        if embed_norm == 'layernorm':
            self.embed_norm_layer = nn.LayerNorm(embed_dim)
        else:
            self.embed_norm_layer = None

        # Prediction heads
        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)  # embed + action → reward
        self.done_pred = nn.Linear(embed_dim + n_actions, 1)    # embed + action → done probability

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        print(f"World Model initialized. Input shape: {observation_shape}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Embed normalization: {embed_norm}")
        print(f"  Dynamics: embed + action → next_embed")
        print(f"  Prediction heads: reward, done")


    def normalize_embedding(self, embed):
        """
        Apply normalization to embeddings for stability.

        Args:
            embed: (..., embed_dim) embeddings

        Returns:
            normalized embeddings with same shape
        """
        if self.embed_norm_type is None:
            return embed
        elif self.embed_norm_type == 'layernorm':
            return self.embed_norm_layer(embed)
        elif self.embed_norm_type == 'tanh':
            return torch.tanh(embed)
        elif self.embed_norm_type == 'l2':
            # L2 normalize to unit sphere
            return F.normalize(embed, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown embed_norm_type: {self.embed_norm_type}")

    def encode(self, obs):
        # If obs is [B, C, H, W], add sequence dimension -> [B, 1, C, H, W]
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        batch_size, sequence_length = obs.shape[:2]
        obs_flat = obs.view(batch_size * sequence_length, *obs.shape[2:])
        embed_flat = self.encoder(obs_flat)

        # Normalize embeddings
        embed_flat = self.normalize_embedding(embed_flat)

        embeds = embed_flat.view(batch_size, sequence_length, -1)

        return embeds

    def decode(self, embeds):
        return self.decoder(embeds)

    def imagine_step(self, embed, action_onehot):
        """
        Imagination step in latent space (no decoding).

        Args:
            embed: (B, embed_dim) current state embedding
            action_onehot: (B, n_actions) one-hot encoded action

        Returns:
            next_embed: (B, embed_dim) predicted next state embedding
            reward: (B, 1) predicted reward
            done: (B, 1) predicted done probability
        """
        # Predict next embedding and normalize it
        next_embed = self.dynamics(embed, action_onehot)
        next_embed = self.normalize_embedding(next_embed)

        # Predict reward and done
        embed_action = torch.cat([embed, action_onehot], dim=-1)
        reward = torch.tanh(self.reward_pred(embed_action))
        done = torch.sigmoid(self.done_pred(embed_action))

        return next_embed, reward, done

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
        recon, embeds, next_embed_pred, reward_pred, done_pred = self.forward(obs_normalized, action_onehot)

        # === 1. Reconstruction Loss ===
        l1_loss = F.l1_loss(recon, obs_normalized)
        structural_loss = ssim_loss(recon, obs_normalized)
        edge_loss = gradient_loss(recon, obs_normalized)
        recon_loss = l1_loss + 0.2 * structural_loss + 0.1 * edge_loss

        # === 2. Dynamics Loss ===
        # Encode next observation to get target embedding
        next_embeds = self.encode(next_obs_normalized)  # (B, 1, embed_dim)
        next_embed_target = next_embeds.view(-1, next_embeds.shape[-1])  # (B, embed_dim)

        # MSE between predicted and actual next embedding
        # Up-weight transitions where reward fired (brick hits, life losses) — 2× emphasis
        per_sample_dynamics = F.mse_loss(next_embed_pred, next_embed_target.detach(), reduction='none').mean(dim=1)
        reward_weights = 1.0 + rewards.float().abs()  # {0-reward: 1.0, nonzero: 2.0}
        dynamics_loss = (reward_weights * per_sample_dynamics).mean()

        # === 3. Reward Loss ===
        # Reward is in range [-1, 0, 1] after life penalty
        reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards.float())

        # Supervise dynamics output with same reward head — forces dynamics to
        # preserve reward-relevant features (ball position, brick proximity) through rollout steps
        next_embed_action = torch.cat([next_embed_pred, action_onehot], dim=-1)
        next_reward_pred = torch.tanh(self.reward_pred(next_embed_action))
        dynamics_reward_loss = F.mse_loss(next_reward_pred.squeeze(-1), rewards.float())

        # === 4. Done Loss ===
        # Binary classification
        done_loss = F.binary_cross_entropy(done_pred.squeeze(-1), dones.float())

        # === Combined Loss ===
        combined_loss = (
            1.0 * recon_loss +
            1.0 * dynamics_loss +
            2.0 * reward_loss +
            2.0 * dynamics_reward_loss +
            0.5 * done_loss
        )

        return combined_loss, {
            "total": combined_loss.item(),
            "recon": recon_loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward": reward_loss.item(),
            "dynamics_reward": dynamics_reward_loss.item(),
            "done": done_loss.item(),
            "l1": l1_loss.item(),
            "ssim": structural_loss.item(),
            "edge": edge_loss.item(),
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
            done_pred: (B, 1) predicted done probability
        """
        # Encode observation to latent state
        embeds = self.encode(obs)  # (B, 1, embed_dim)

        # Decode for reconstruction
        embeds_flat = embeds.view(-1, embeds.shape[-1])  # (B, embed_dim)
        recon = self.decode(embeds_flat)  # (B, C, H, W)

        # Flatten embeddings for predictions
        embed = embeds_flat  # (B, embed_dim)

        # Predict next embedding using dynamics model and normalize it
        next_embed_pred = self.dynamics(embed, action_onehot)  # (B, embed_dim)
        next_embed_pred = self.normalize_embedding(next_embed_pred)

        # Predict reward from current state + action
        embed_action = torch.cat([embed, action_onehot], dim=-1)
        reward_pred = torch.tanh(self.reward_pred(embed_action))  # (B, 1) in [-1, 1]

        # Predict done from current state + action
        done_pred = torch.sigmoid(self.done_pred(embed_action))  # (B, 1) in [0, 1]

        return recon, embeds, next_embed_pred, reward_pred, done_pred
    



