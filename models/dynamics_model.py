import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel


class DynamicsModel(BaseModel):
    """
    Predicts next latent state from current latent state + action.

    This is the core of the world model - it learns the dynamics of the environment
    in latent space (embeddings) rather than pixel space.

    Input: embed_t (B, embed_dim) + action (B, n_actions as one-hot)
    Output: embed_{t+1} (B, embed_dim)
    """

    def __init__(self, embed_dim=1024, n_actions=4, hidden_dim=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        # MLP to predict next embedding
        self.net = nn.Sequential(
            nn.Linear(embed_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        print(f"DynamicsModel initialized:")
        print(f"  Input: embed({embed_dim}) + action({n_actions}) = {embed_dim + n_actions}")
        print(f"  Hidden: {hidden_dim}")
        print(f"  Output: next_embed({embed_dim})")

    def forward(self, embed, action_onehot):
        """
        Predict next embedding given current embedding and action.

        Args:
            embed: (B, embed_dim) current latent state
            action_onehot: (B, n_actions) one-hot encoded action

        Returns:
            next_embed: (B, embed_dim) predicted next latent state
        """
        # Concatenate embedding and action
        x = torch.cat([embed, action_onehot], dim=-1)

        # Predict next embedding
        next_embed = self.net(x)

        return next_embed
