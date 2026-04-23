import os
import gymnasium as gym
import torch
from buffer import ReplayBuffer
from utils import display_stacked_obs
from models.world_model import WorldModel
from models.q_model import QModel
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import random


def get_wm_q_ratio(episode):
    """Dynamic world model to Q-model training ratio based on episode.

    Keeps world model training strong throughout to track the evolving
    data distribution. Never drops below a meaningful WM update rate
    to prevent world model degradation during Q-heavy phases.
    """
    if episode < 25:
        return [4, 0]   # WM-only: build foundation
    elif episode < 100:
        return [3, 1]   # Start Q training
    elif episode < 250:
        return [2, 2]   # Balanced: let WM stabilize
    else:
        return [2, 3]   # Q-focused but WM stays strong


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 10000,
                       world_model_batch_size: int = 8,
                       target_update_interval: int = 10000) -> None:
        self.env    = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        initial_obs, _ = self.env.reset()
        processed_obs  = self.process_observation(initial_obs)
        n_actions      = int(self.env.action_space.n)  # type: ignore[attr-defined]

        self.memory = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=processed_obs.shape,
            n_actions=n_actions,
            input_device=self.device,
            output_device=self.device,
        )

        self.world_model = WorldModel(
            observation_shape=processed_obs.shape,
            embed_dim=1024,
            rssm_hidden_dim=1024,
            n_actions=n_actions,
            embed_norm='layernorm',
        ).to(self.device)

        print(f"Observation shape: {processed_obs.shape}")

        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=0.0001
        )
        self.world_model_batch_size = world_model_batch_size

        # Q-model operates on RSSM hidden states (same dim as rssm_hidden_dim)
        self.q_model = QModel(
            action_dim=n_actions,
            hidden_dim=256,
            embed_dim=self.world_model.rssm_hidden_dim,
        ).to(self.device)

        self.target_q_model = QModel(
            action_dim=n_actions,
            hidden_dim=256,
            embed_dim=self.world_model.rssm_hidden_dim,
        ).to(self.device)

        self.q_model_optimizer = torch.optim.Adam(
            self.q_model.parameters(), lr=0.0001
        )

        self.target_update_interval = target_update_interval
        self.gamma                  = 0.99

        # Real-environment exploration
        self.epsilon         = 1.0
        self.min_epsilon     = 0.1
        self.epsilon_decay   = 0.98

        # Imagination exploration
        self.imagine_epsilon       = 1.0
        self.imagine_min_epsilon   = 0.1
        self.imagine_epsilon_decay = 0.99

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Observation utilities
    # ------------------------------------------------------------------

    def normalize_observation(self, obs):
        return obs / 255.0

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        obs = torch.from_numpy(obs).permute(2, 0, 1)
        return obs

    # ------------------------------------------------------------------
    # Imagination rollout (pure latent space, RSSM hidden state)
    # ------------------------------------------------------------------

    def imagine_trajectory(self, batch_size, horizon):
        """
        Imagine batch_size parallel trajectories for horizon steps in latent space.

        Starting observations are sampled from the replay buffer and encoded
        into initial RSSM hidden states. From there, each step uses the
        prior_predictor (no real observations) to advance the hidden state —
        the same distribution the RSSM trained on.

        Returns flattened tensors of (batch_size * horizon, ...) for Q-training.
        """
        n_actions = int(self.env.action_space.n)  # type: ignore[attr-defined]

        # Sample starting observations and encode them to initial hidden states
        sampled_obs, _, _, _, _ = self.memory.sample_buffer(batch_size)
        sampled_obs_normalized  = self.normalize_observation(sampled_obs)

        with torch.no_grad():
            current_hidden_states = self.world_model.encode_observation_to_hidden(
                sampled_obs_normalized
            )  # (batch_size, rssm_hidden_dim)

        # Accumulators for all rollout steps
        all_hidden_states      = []
        all_actions            = []
        all_rewards            = []
        all_next_hidden_states = []
        all_dones              = []

        for _ in range(horizon):
            with torch.no_grad():
                # Select actions using Q-model on current hidden states (epsilon-greedy)
                q_values     = self.q_model(current_hidden_states)
                best_actions = q_values.argmax(dim=1)

                random_actions  = torch.randint(0, n_actions, (batch_size,), device=self.device)
                is_exploring    = (torch.rand(batch_size, device=self.device) < self.imagine_epsilon)
                selected_actions = torch.where(is_exploring, random_actions, best_actions)

                action_onehot = F.one_hot(selected_actions, num_classes=n_actions).float()

                # Advance the world model using the prior (no real observation)
                next_hidden_states, rewards, dones = self.world_model.imagine_step(
                    current_hidden_states, action_onehot
                )

                all_hidden_states.append(current_hidden_states)
                all_actions.append(selected_actions)
                all_rewards.append(rewards.squeeze(-1))
                all_next_hidden_states.append(next_hidden_states)
                all_dones.append((dones.squeeze(-1) > 0.5).float())

                current_hidden_states = next_hidden_states

        # Flatten along batch × horizon
        hidden_states      = torch.cat(all_hidden_states,      dim=0)   # (B*H, rssm_hidden_dim)
        actions            = torch.cat(all_actions,            dim=0)   # (B*H,)
        rewards            = torch.cat(all_rewards,            dim=0)   # (B*H,)
        next_hidden_states = torch.cat(all_next_hidden_states, dim=0)   # (B*H, rssm_hidden_dim)
        dones              = torch.cat(all_dones,              dim=0)   # (B*H,)

        return hidden_states, actions, rewards, next_hidden_states, dones

    # ------------------------------------------------------------------
    # World model training (sequence-based for RSSM BPTT)
    # ------------------------------------------------------------------

    def train_world_model(self, epochs, batch_size, seq_len):
        """
        Train the world model on sequences of real transitions.

        Sequences are required (rather than individual samples) so that BPTT
        can propagate gradients through the GRU hidden state across multiple
        timesteps. This trains the RSSM to carry information coherently
        through time — the property that makes imagination reliable.
        """
        if not self.memory.can_sample_sequences(batch_size, seq_len):
            # Not enough consecutive transitions yet — skip silently
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        total_loss         = 0.0
        total_recon_loss   = 0.0
        total_prior_loss   = 0.0
        total_reward_loss  = 0.0
        total_done_loss    = 0.0
        total_l1_loss      = 0.0
        total_ssim_loss    = 0.0
        total_edge_loss    = 0.0

        for _ in range(epochs):
            obs_seq, action_seq, reward_seq, _, done_seq = self.memory.sample_sequences(
                batch_size=batch_size,
                seq_len=seq_len,
            )

            # obs_seq is float32 from the buffer; convert to uint8 range expected by compute_loss
            obs_seq_uint8 = obs_seq.to(torch.uint8)

            loss, loss_dict = self.world_model.compute_loss(
                obs_seq    = obs_seq_uint8,
                action_seq = action_seq,
                reward_seq = reward_seq,
                done_seq   = done_seq,
            )

            self.world_model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()

            total_loss        += loss_dict["total"]
            total_recon_loss  += loss_dict["recon"]
            total_prior_loss  += loss_dict["prior"]
            total_reward_loss += loss_dict["reward"]
            total_done_loss   += loss_dict["done"]
            total_l1_loss     += loss_dict["l1"]
            total_ssim_loss   += loss_dict["ssim"]
            total_edge_loss   += loss_dict["edge"]

        avg_total        = total_loss        / epochs
        avg_recon_loss   = total_recon_loss  / epochs
        avg_prior_loss   = total_prior_loss  / epochs
        avg_reward_loss  = total_reward_loss / epochs
        avg_done_loss    = total_done_loss   / epochs
        avg_l1_loss      = total_l1_loss     / epochs
        avg_ssim_loss    = total_ssim_loss   / epochs
        avg_edge_loss    = total_edge_loss   / epochs

        return (avg_total, avg_reward_loss, avg_done_loss, avg_recon_loss,
                avg_prior_loss, avg_l1_loss, avg_ssim_loss, avg_edge_loss)

    # ------------------------------------------------------------------
    # Q-model training on imagined trajectories
    # ------------------------------------------------------------------

    def train_q_model_on_imagination(self, horizon, batch_size, epochs=1):
        """
        Train the Q-model on imagined trajectories in RSSM hidden-state space.

        Uses Double DQN targets:
          - Online network selects the best next action
          - Target network evaluates that action's Q-value
        This reduces the overestimation bias of standard DQN.
        """
        total_q_loss      = 0.0
        total_imag_reward = 0.0

        for _ in range(epochs):
            hidden_states, actions, rewards, next_hidden_states, dones = \
                self.imagine_trajectory(batch_size, horizon)

            actions = actions.unsqueeze(1).long()
            rewards = rewards.unsqueeze(1)
            dones   = dones.unsqueeze(1).float()

            # Current Q-values for the actions taken
            q_values     = self.q_model(hidden_states)
            q_sa         = q_values.gather(1, actions)

            with torch.no_grad():
                # Double DQN: online network picks action, target network evaluates it
                best_next_actions = self.q_model(next_hidden_states).argmax(dim=1, keepdim=True)
                next_q_values     = self.target_q_model(next_hidden_states).gather(1, best_next_actions)
                td_targets        = rewards + (1 - dones) * self.gamma * next_q_values

            q_loss = F.mse_loss(q_sa, td_targets)

            self.q_model_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
            self.q_model_optimizer.step()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

            total_q_loss      += q_loss.item()
            total_imag_reward += rewards.mean().item()
            self.total_steps  += 1

        return total_q_loss / epochs, total_imag_reward / epochs

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_reconstruction(self, num_samples=4, filename="reconstruction_test.png"):
        """Compare original vs reconstructed observations and save to disk."""
        if not self.memory.can_sample(num_samples):
            return

        obs, _, _, _, _ = self.memory.sample_buffer(num_samples)
        obs_normalized  = obs.float() / 255.0

        with torch.no_grad():
            reconstructed_obs, _ = self.world_model.forward(obs_normalized)

        viz_pairs = []
        for i in range(num_samples):
            viz_pairs.append((f"original_{i}",     obs_normalized[i].cpu()))
            viz_pairs.append((f"reconstructed_{i}", reconstructed_obs[i].cpu()))

        display_stacked_obs(viz_pairs, filename, num_frames=1)
        print(f"Saved reconstruction comparison to {filename}")

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save(self):
        self.world_model.save_the_model("world_model", verbose=True)
        self.q_model.save_the_model("q_model", verbose=True)

    def load(self):
        self.world_model.load_the_model("world_model", device=self.device)
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_the_model("q_model", device=self.device)

    # ------------------------------------------------------------------
    # Test (greedy policy in real environment)
    # ------------------------------------------------------------------

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode_idx in range(episodes):
            obs, _         = self.env.reset()
            obs            = self.process_observation(obs)
            done           = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_normalized = obs.unsqueeze(0).float().to(self.device) / 255.0
                    hidden_state   = self.world_model.encode_observation_to_hidden(obs_normalized)
                    action         = self.q_model(hidden_state).argmax(dim=1).item()

                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs        = self.process_observation(next_obs)
                done            = term or trunc
                episode_reward += float(reward)
                obs             = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode_idx} | reward: {episode_reward:.1f}")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward:.1f}")
        self.q_model.train()
        return total_rewards

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, episodes=1, offline_training_epochs=1, batch_size=1,
              num_batches=1, wm_batch_size=1, wm_seq_len=16, imagination_steps=None):

        rollout_steps = imagination_steps if imagination_steps is not None else batch_size

        run_tag = (
            f'world_model_rssm'
            f'_ote{offline_training_epochs}'
            f'_bs{batch_size}'
            f'_wmbs{wm_batch_size}'
            f'_seqlen{wm_seq_len}'
            f'_rollout{rollout_steps}'
            f'_buf{self.memory.mem_size}'
        )
        log_dir = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'
        writer  = SummaryWriter(log_dir)

        for episode in range(episodes):

            obs, _ = self.env.reset()
            obs    = self.process_observation(obs)

            done           = False
            episode_reward = 0.0
            episode_steps  = 0

            # --- Collect one episode of real experience ---
            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_normalized = obs.unsqueeze(0).float().to(self.device) / 255.0
                        hidden_state   = self.world_model.encode_observation_to_hidden(obs_normalized)
                        action         = self.q_model(hidden_state).argmax(dim=1).item()

                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                done     = term or trunc

                self.memory.store_transition(obs, action, reward, next_obs, done)

                episode_reward += float(reward)
                episode_steps  += 1
                obs             = next_obs

            # Decay exploration rates
            self.epsilon        = max(self.min_epsilon,        self.epsilon        * self.epsilon_decay)
            self.imagine_epsilon = max(self.imagine_min_epsilon, self.imagine_epsilon * self.imagine_epsilon_decay)

            print(f"Episode {episode} | reward: {episode_reward:.1f} | "
                  f"epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            # --- Interleaved training with dynamic WM/Q ratio ---
            current_ratio = get_wm_q_ratio(episode)

            total_combined_loss  = 0.0
            total_reward_loss    = 0.0
            total_done_loss      = 0.0
            total_recon_loss     = 0.0
            total_prior_loss     = 0.0
            total_l1_loss        = 0.0
            total_ssim_loss      = 0.0
            total_edge_loss      = 0.0
            total_q_loss         = 0.0
            total_imag_reward    = 0.0
            wm_updates           = 0
            q_updates            = 0

            for _ in range(offline_training_epochs):
                # World model updates (sequence-based RSSM training)
                for _ in range(current_ratio[0]):
                    (combined_loss, reward_loss, done_loss, recon_loss,
                     prior_loss, l1_loss, ssim_loss, edge_loss) = self.train_world_model(
                        epochs=1, batch_size=wm_batch_size, seq_len=wm_seq_len
                    )
                    total_combined_loss += combined_loss
                    total_reward_loss   += reward_loss
                    total_done_loss     += done_loss
                    total_recon_loss    += recon_loss
                    total_prior_loss    += prior_loss
                    total_l1_loss       += l1_loss
                    total_ssim_loss     += ssim_loss
                    total_edge_loss     += edge_loss
                    wm_updates          += 1

                # Q-model updates on imagined trajectories
                for _ in range(current_ratio[1]):
                    q_loss, imag_reward = self.train_q_model_on_imagination(
                        rollout_steps, batch_size, epochs=1
                    )
                    total_q_loss      += q_loss
                    total_imag_reward += imag_reward
                    q_updates         += 1

            # --- Average losses ---
            safe_wm_denom = wm_updates if wm_updates > 0 else 1
            safe_q_denom  = q_updates  if q_updates  > 0 else 1

            avg_combined_loss = total_combined_loss / safe_wm_denom
            avg_reward_loss   = total_reward_loss   / safe_wm_denom
            avg_done_loss     = total_done_loss     / safe_wm_denom
            avg_recon_loss    = total_recon_loss    / safe_wm_denom
            avg_prior_loss    = total_prior_loss    / safe_wm_denom
            avg_l1_loss       = total_l1_loss       / safe_wm_denom
            avg_ssim_loss     = total_ssim_loss     / safe_wm_denom
            avg_edge_loss     = total_edge_loss     / safe_wm_denom
            avg_q_loss        = total_q_loss        / safe_q_denom

            # --- TensorBoard logging ---
            writer.add_scalar("World Model/combined_loss",      avg_combined_loss, episode)
            writer.add_scalar("World Model/reward_loss",        avg_reward_loss,   episode)
            writer.add_scalar("World Model/done_loss",          avg_done_loss,     episode)
            writer.add_scalar("World Model/reconstruction_loss", avg_recon_loss,   episode)
            writer.add_scalar("World Model/prior_loss",         avg_prior_loss,    episode)
            writer.add_scalar("Reconstruction/l1_loss",         avg_l1_loss,       episode)
            writer.add_scalar("Reconstruction/ssim_loss",       avg_ssim_loss,     episode)
            writer.add_scalar("Reconstruction/edge_loss",       avg_edge_loss,     episode)
            writer.add_scalar("Train/wm_updates_per_episode",   wm_updates,        episode)
            writer.add_scalar("Train/q_updates_per_episode",    q_updates,         episode)
            writer.add_scalar("Train/target_update_interval",   self.target_update_interval, episode)
            writer.add_scalar("Train/updates_per_cycle_wm",     current_ratio[0],  episode)
            writer.add_scalar("Train/updates_per_cycle_q",      current_ratio[1],  episode)
            writer.add_scalar("Train/episode_reward",           episode_reward,    episode)
            writer.add_scalar("Train/epsilon",                  self.epsilon,      episode)
            writer.add_scalar("Train/imagine_epsilon",          self.imagine_epsilon, episode)
            writer.add_scalar("Train/avg_q_loss",               avg_q_loss,        episode)

            if q_updates > 0:
                avg_imag_reward        = total_imag_reward / q_updates
                real_reward_per_step   = episode_reward / episode_steps if episode_steps > 0 else 0.0
                writer.add_scalar("Imagination/mean_reward_per_step", avg_imag_reward,      episode)
                writer.add_scalar("Imagination/real_reward_per_step", real_reward_per_step, episode)
                writer.add_scalar("Imagination/vs_real_reward_diff",
                                  avg_imag_reward - real_reward_per_step, episode)

            if episode % 100 == 0:
                print(f"Completed episode {episode} — reward loss: {avg_reward_loss:.4f} | "
                      f"prior loss: {avg_prior_loss:.4f}")

            if episode % 10 == 0:
                self.evaluate_reconstruction(num_samples=4, filename="reconstruction_test.png")
                self.save()
