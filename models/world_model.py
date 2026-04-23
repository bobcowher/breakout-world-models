import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder, Decoder
from models.base import BaseModel
from models.ssim_loss import ssim_loss
from models.rssm import RSSM


def gradient_loss(predicted_image, target_image):
    """
    Compute edge/gradient loss between predicted and target images.
    Preserves high-contrast edges and small objects (like the ball).
    Computes L1 loss on horizontal and vertical image gradients.

    Args:
        predicted_image : (batch_size, channels, H, W)
        target_image    : (batch_size, channels, H, W)

    Returns:
        scalar gradient loss
    """
    pred_dx   = predicted_image[:, :, :, 1:] - predicted_image[:, :, :, :-1]
    target_dx = target_image[:, :, :, 1:]    - target_image[:, :, :, :-1]

    pred_dy   = predicted_image[:, :, 1:, :] - predicted_image[:, :, :-1, :]
    target_dy = target_image[:, :, 1:, :]    - target_image[:, :, :-1, :]

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=1024, rssm_hidden_dim=1024,
                 n_actions=4, embed_norm='layernorm'):
        """
        Args:
            observation_shape : (C, H, W) of the input observations
            embed_dim         : encoder output dimension (fed into RSSM)
            rssm_hidden_dim   : GRU hidden state dimension; used for decoding,
                                reward/done prediction, and Q-learning
            n_actions         : number of discrete actions
            embed_norm        : normalization applied to raw encoder output
        """
        super().__init__()

        # Encoder: pixels → flat embedding
        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)

        # Decoder: RSSM hidden state → reconstructed pixels
        # Uses rssm_hidden_dim so decoding comes from the recurrent state,
        # not the raw encoder output.
        self.decoder = Decoder(
            observation_shape=observation_shape,
            embed_dim=rssm_hidden_dim,
            conv_output_shape=self.encoder.get_output_shape(),
            conv_channels=self.encoder.get_conv_channels(),
        )

        # RSSM: replaces the stateless MLP dynamics model
        self.rssm = RSSM(
            encoder_dim=embed_dim,
            n_actions=n_actions,
            hidden_dim=rssm_hidden_dim,
        )

        # Embedding normalization on raw encoder output (before feeding RSSM)
        self.embed_norm_type = embed_norm
        if embed_norm == 'layernorm':
            self.embed_norm_layer = nn.LayerNorm(embed_dim)
        else:
            self.embed_norm_layer = None

        # Prediction heads — operate on the RSSM hidden state + action
        self.reward_pred = nn.Linear(rssm_hidden_dim + n_actions, 1)
        self.done_pred   = nn.Linear(rssm_hidden_dim + n_actions, 1)

        self.embed_dim       = embed_dim
        self.rssm_hidden_dim = rssm_hidden_dim
        self.n_actions       = n_actions

        print(f"WorldModel initialized. Input shape: {observation_shape}")
        print(f"  Encoder embed dim:  {embed_dim}")
        print(f"  RSSM hidden dim:    {rssm_hidden_dim}")
        print(f"  Embed norm:         {embed_norm}")
        print(f"  Dynamics:           RSSM (GRU + prior predictor)")
        print(f"  Prediction heads:   reward, done")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def normalize_embedding(self, raw_embed):
        """Apply normalization to raw encoder output."""
        if self.embed_norm_type is None:
            return raw_embed
        elif self.embed_norm_type == 'layernorm':
            assert self.embed_norm_layer is not None
            return self.embed_norm_layer(raw_embed)
        elif self.embed_norm_type == 'tanh':
            return torch.tanh(raw_embed)
        elif self.embed_norm_type == 'l2':
            return F.normalize(raw_embed, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown embed_norm_type: {self.embed_norm_type}")

    def encode(self, obs):
        """
        Encode observations to normalized embeddings.

        Args:
            obs : (batch_size, C, H, W) or (batch_size, 1, C, H, W) float in [0,1]

        Returns:
            embeddings : (batch_size, 1, embed_dim)
        """
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        batch_size, seq_len = obs.shape[:2]
        obs_flat            = obs.view(batch_size * seq_len, *obs.shape[2:])
        embed_flat          = self.encoder(obs_flat)
        embed_flat          = self.normalize_embedding(embed_flat)

        return embed_flat.view(batch_size, seq_len, -1)

    def decode(self, hidden_state):
        """
        Decode RSSM hidden state to reconstructed observation.

        Args:
            hidden_state : (batch_size, rssm_hidden_dim)

        Returns:
            reconstructed_obs : (batch_size, C, H, W) in [0, 1]
        """
        return self.decoder(hidden_state)

    # ------------------------------------------------------------------
    # Warm-up: initialise hidden state from a real observation
    # ------------------------------------------------------------------

    def encode_observation_to_hidden(self, obs_normalized):
        """
        Produce an initial RSSM hidden state from a single real observation.
        Used to seed the hidden state before imagination rollouts.

        One RSSM step is run with a NOOP action so the GRU has seen at least
        one real observation before imagination begins.

        Args:
            obs_normalized : (batch_size, C, H, W) float in [0, 1]

        Returns:
            initial_hidden_state : (batch_size, rssm_hidden_dim)
        """
        batch_size     = obs_normalized.shape[0]
        encoder_output = self.encode(obs_normalized).squeeze(1)  # (batch_size, embed_dim)

        zero_hidden = self.rssm.get_initial_hidden_state(batch_size, obs_normalized.device)

        # NOOP action (index 0) as one-hot
        noop_action_onehot        = torch.zeros(batch_size, self.n_actions, device=obs_normalized.device)
        noop_action_onehot[:, 0] = 1.0

        initial_hidden_state = self.rssm.step_with_observation(
            prev_hidden_state=zero_hidden,
            encoder_output=encoder_output,
            action_onehot=noop_action_onehot,
        )
        return initial_hidden_state

    # ------------------------------------------------------------------
    # Imagination step (used by agent and visualizer)
    # ------------------------------------------------------------------

    def imagine_step(self, hidden_state, action_onehot):
        """
        Advance the world model by one imagination step (no real observation).

        Args:
            hidden_state  : (batch_size, rssm_hidden_dim)
            action_onehot : (batch_size, n_actions)

        Returns:
            next_hidden_state : (batch_size, rssm_hidden_dim)
            reward            : (batch_size, 1)  in [-1, 1]
            done              : (batch_size, 1)  probability in [0, 1]
        """
        next_hidden_state, _ = self.rssm.step_with_prior(hidden_state, action_onehot)

        hidden_action = torch.cat([next_hidden_state, action_onehot], dim=-1)
        reward        = torch.tanh(self.reward_pred(hidden_action))
        done          = torch.sigmoid(self.done_pred(hidden_action))

        return next_hidden_state, reward, done

    # ------------------------------------------------------------------
    # Sequence training loss
    # ------------------------------------------------------------------

    def compute_loss(self, obs_seq, action_seq, reward_seq, done_seq):
        """
        Compute world model losses over a sequence of real transitions.

        The first timestep is used as a warm-up to initialise the hidden state
        from a real observation (gradients do not flow through this step).
        Losses are then computed for timesteps 1 … seq_len-1.

        For each training timestep t the model:
          1. Predicts what the encoder would produce (prior_predictor → prior_loss)
          2. Decodes the hidden state to reconstruct obs_t (recon_loss)
          3. Predicts the reward and done signal (reward_loss, done_loss)
          4. Updates the hidden state with the REAL encoder output (RSSM step)
          5. Resets hidden state to zero at episode boundaries (done=True)

        Args:
            obs_seq    : (batch_size, seq_len, C, H, W)  uint8
            action_seq : (batch_size, seq_len)            int64
            reward_seq : (batch_size, seq_len)            float
            done_seq   : (batch_size, seq_len)            bool

        Returns:
            combined_loss : scalar
            loss_dict     : dict of individual loss components
        """
        batch_size, seq_len = obs_seq.shape[:2]
        device              = obs_seq.device

        obs_normalized = obs_seq.float() / 255.0  # (B, T, C, H, W)

        # --- Warm-up step (no gradient): initialise hidden state from obs[0] ---
        with torch.no_grad():
            encoder_output_0   = self.encode(obs_normalized[:, 0]).squeeze(1)
            noop_action_onehot = torch.zeros(batch_size, self.n_actions, device=device)
            noop_action_onehot[:, 0] = 1.0
            hidden_state = self.rssm.step_with_observation(
                prev_hidden_state=self.rssm.get_initial_hidden_state(batch_size, device),
                encoder_output=encoder_output_0,
                action_onehot=noop_action_onehot,
            )

        # --- Accumulate losses for steps 1 … seq_len-1 ---
        total_recon_loss  = 0.0
        total_reward_loss = 0.0
        total_done_loss   = 0.0
        total_prior_loss  = 0.0
        total_l1_loss     = 0.0
        total_ssim_loss   = 0.0
        total_edge_loss   = 0.0

        num_training_steps = seq_len - 1  # exclude warm-up step

        for step_idx in range(1, seq_len):
            obs_t          = obs_normalized[:, step_idx]              # (B, C, H, W)
            action_t       = action_seq[:, step_idx]                  # (B,)
            reward_t       = reward_seq[:, step_idx]                  # (B,)
            done_t         = done_seq[:, step_idx]                    # (B,)

            action_onehot_t = F.one_hot(action_t.long(), num_classes=self.n_actions).float()

            # Encode real observation (used for prior supervision and RSSM update)
            encoder_output_t = self.encode(obs_t).squeeze(1)  # (B, embed_dim)

            # 1. Prior loss: train prior_predictor to predict encoder output
            #    from the current hidden state alone (the imagination signal)
            predicted_encoder_output = self.rssm.prior_predictor(hidden_state)
            prior_loss_t = F.mse_loss(predicted_encoder_output, encoder_output_t.detach())

            # 2. Reconstruction loss: decode hidden state → reconstruct obs_t
            reconstructed_obs = self.decode(hidden_state)
            l1_loss_t         = F.l1_loss(reconstructed_obs, obs_t)
            ssim_loss_t       = ssim_loss(reconstructed_obs, obs_t)
            edge_loss_t       = gradient_loss(reconstructed_obs, obs_t)
            recon_loss_t      = l1_loss_t + 0.2 * ssim_loss_t + 0.1 * edge_loss_t

            # 3. Reward and done prediction from hidden state + action
            hidden_action_concat = torch.cat([hidden_state, action_onehot_t], dim=-1)
            reward_pred_t        = torch.tanh(self.reward_pred(hidden_action_concat))
            done_pred_t          = torch.sigmoid(self.done_pred(hidden_action_concat))

            reward_loss_t = F.mse_loss(reward_pred_t.squeeze(-1), reward_t.float())
            done_loss_t   = F.binary_cross_entropy(done_pred_t.squeeze(-1), done_t.float())

            # Accumulate
            total_prior_loss  += prior_loss_t
            total_recon_loss  += recon_loss_t
            total_reward_loss += reward_loss_t
            total_done_loss   += done_loss_t
            total_l1_loss     += l1_loss_t
            total_ssim_loss   += ssim_loss_t
            total_edge_loss   += edge_loss_t

            # 4. Update hidden state with the REAL encoder output (training step)
            hidden_state = self.rssm.step_with_observation(
                prev_hidden_state=hidden_state,
                encoder_output=encoder_output_t,
                action_onehot=action_onehot_t,
            )

            # 5. Reset hidden state to zero at episode boundaries
            done_mask    = done_t.float().unsqueeze(-1)   # (B, 1)
            hidden_state = hidden_state * (1.0 - done_mask)

        # --- Average losses over training steps ---
        avg_recon_loss  = total_recon_loss  / num_training_steps
        avg_reward_loss = total_reward_loss / num_training_steps
        avg_done_loss   = total_done_loss   / num_training_steps
        avg_prior_loss  = total_prior_loss  / num_training_steps
        avg_l1_loss     = total_l1_loss     / num_training_steps
        avg_ssim_loss   = total_ssim_loss   / num_training_steps
        avg_edge_loss   = total_edge_loss   / num_training_steps

        combined_loss = (
            1.0 * avg_recon_loss  +
            2.0 * avg_reward_loss +
            0.5 * avg_done_loss   +
            1.0 * avg_prior_loss
        )

        loss_dict = {
            "total":   combined_loss.item(),
            "recon":   avg_recon_loss.item(),
            "reward":  avg_reward_loss.item(),
            "done":    avg_done_loss.item(),
            "prior":   avg_prior_loss.item(),
            "l1":      avg_l1_loss.item(),
            "ssim":    avg_ssim_loss.item(),
            "edge":    avg_edge_loss.item(),
        }

        return combined_loss, loss_dict

    # ------------------------------------------------------------------
    # Legacy forward (used by evaluate_reconstruction in agent.py)
    # ------------------------------------------------------------------

    def forward(self, obs_normalized, action_onehot=None):
        """
        Single-step forward pass for evaluation and visualisation.
        Not used during sequence training.

        Returns hidden_state and reconstruction so callers can compare
        real vs reconstructed observations.
        """
        hidden_state      = self.encode_observation_to_hidden(obs_normalized)
        reconstructed_obs = self.decode(hidden_state)
        return reconstructed_obs, hidden_state
