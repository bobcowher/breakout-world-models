import torch
import torch.nn as nn
from models.base import BaseModel


class RSSM(BaseModel):
    """
    Recurrent State Space Model (deterministic variant, no stochastic latents).

    Replaces the stateless MLP dynamics model with a GRU that maintains a
    hidden state across timesteps. This fixes the core training/imagination
    distribution mismatch: the GRU is always trained on its own hidden states
    (via BPTT), so imagination rollouts use the same distribution they trained on.

    Two operating modes:

      Training (real observation available):
        hidden_state_t = GRU(hidden_state_{t-1}, concat(encoder_output_t, action_t))

      Imagination (no real observation available):
        predicted_encoder_output = prior_predictor(hidden_state_{t-1})
        hidden_state_t = GRU(hidden_state_{t-1}, concat(predicted_encoder_output, action_t))

    The prior_predictor is a small MLP supervised to match the real encoder output.
    It is what allows the GRU to keep running coherently without real observations.
    """

    def __init__(self, encoder_dim, n_actions, hidden_dim):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.n_actions   = n_actions
        self.hidden_dim  = hidden_dim

        # Core recurrent cell: processes (encoder_output, action) → next hidden state
        self.gru_cell = nn.GRUCell(
            input_size=encoder_dim + n_actions,
            hidden_size=hidden_dim,
        )

        # Prior predictor: estimates what the encoder would have produced,
        # given only the current hidden state. Used during imagination in place
        # of a real encoder output.
        self.prior_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoder_dim),
        )

        print(f"RSSM initialized:")
        print(f"  Encoder dim (GRU obs input): {encoder_dim}")
        print(f"  Action dim:                  {n_actions}")
        print(f"  GRU input size:              {encoder_dim + n_actions}")
        print(f"  GRU hidden dim:              {hidden_dim}")

    # ------------------------------------------------------------------
    # Hidden state utilities
    # ------------------------------------------------------------------

    def get_initial_hidden_state(self, batch_size, device):
        """Return a zero hidden state for the start of a new sequence."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    # ------------------------------------------------------------------
    # Training mode: real observation available
    # ------------------------------------------------------------------

    def step_with_observation(self, prev_hidden_state, encoder_output, action_onehot):
        """
        Update the hidden state using a real encoder output (training mode).

        Args:
            prev_hidden_state : (batch_size, hidden_dim)
            encoder_output    : (batch_size, encoder_dim)  — real encoder output
            action_onehot     : (batch_size, n_actions)

        Returns:
            next_hidden_state : (batch_size, hidden_dim)
        """
        gru_input         = torch.cat([encoder_output, action_onehot], dim=-1)
        next_hidden_state = self.gru_cell(gru_input, prev_hidden_state)
        return next_hidden_state

    # ------------------------------------------------------------------
    # Imagination mode: no real observation available
    # ------------------------------------------------------------------

    def step_with_prior(self, prev_hidden_state, action_onehot):
        """
        Update the hidden state using the prior predictor (imagination mode).
        The prior substitutes for the missing real encoder output.

        Args:
            prev_hidden_state        : (batch_size, hidden_dim)
            action_onehot            : (batch_size, n_actions)

        Returns:
            next_hidden_state        : (batch_size, hidden_dim)
            predicted_encoder_output : (batch_size, encoder_dim)  — prior prediction
        """
        predicted_encoder_output = self.prior_predictor(prev_hidden_state)
        gru_input                = torch.cat([predicted_encoder_output, action_onehot], dim=-1)
        next_hidden_state        = self.gru_cell(gru_input, prev_hidden_state)
        return next_hidden_state, predicted_encoder_output
