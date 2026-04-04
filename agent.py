import gymnasium as gym
from collections import deque
import time
import torch
from torch._dynamo.utils import torchscript
from torch.cuda import device_count
from buffer import ReplayBuffer
from utils import display_stacked_obs
from models.world_model import WorldModel
from models.q_model import QModel
import cv2
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import random
from models.perceptual_loss import PerceptualLoss
import numpy as np

def get_wm_q_ratio(episode):
    """Dynamic world model to Q-model training ratio based on episode."""
    if episode < 50:
        return [5, 0]   # WM-only: aggressive world model training
    elif episode < 200:
        return [5, 1]   # WM-heavy: start Q training
    elif episode < 500:
        return [2, 2]   # Balanced
    elif episode < 1500:
        return [1, 5]   # Q-heavy: world model mature
    else:
        return [1, 10]  # Q-dominant: final policy refinement


class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000,
                       world_model_batch_size = 8,
                       target_update_interval = 10000) -> None:
        self.env = env
        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        # print(torch.squeeze(obs).shape)

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024, n_actions=self.env.action_space.n).to(self.device)
        
        print(f"Observation shape: {obs.shape}")

        
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)

        self.world_model_batch_size = world_model_batch_size

        self.next_frame_loss = PerceptualLoss().to(self.device)

        self.q_model = QModel(action_dim=self.env.action_space.n, hidden_dim=256, observation_shape=tuple(obs.shape), obs_stack=self.frame_stack).to(self.device)
        self.target_q_model = QModel(action_dim=self.env.action_space.n, hidden_dim=256, observation_shape=tuple(obs.shape), obs_stack=self.frame_stack).to(self.device)

        self.q_model_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.0001)

        self.target_update_interval = target_update_interval

        self.gamma = 0.99

        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.995
        self.total_steps = 0
    
    def init_frame_stack(self, obs):
        """Call once after env.reset().  Pre-fill both deques."""
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)

    def normalize_observation(self, obs):
        return obs / 255.0

    def process_observation(self, obs, clear_stack=False):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)  

        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST) # shrink to 128
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) # let's do grayscale    
        # obs = torch.from_numpy(obs).permute(2, 0, 1).to(self.device)
        obs = torch.from_numpy(obs)

        if(len(self.frames) < self.frame_stack):
            self.init_frame_stack(obs)
            
        if(clear_stack):
            self.init_frame_stack(obs)

        self.frames.append(obs)

        obs_stacked = torch.stack(list(self.frames), dim=0)

        return obs_stacked


    def imagine_trajectory(self, batch_size, num_batches):
        
        obs, _, _, _, _ = self.memory.sample_buffer(1)

        total_batch_size = batch_size * num_batches

        obs_shape   = obs.shape[1:]  # (4, 96, 96)
        states      = torch.zeros(total_batch_size, *obs_shape)
        actions     = torch.zeros(total_batch_size)
        rewards     = torch.zeros(total_batch_size)
        next_states = torch.zeros(total_batch_size, *obs_shape)
        dones       = torch.zeros(total_batch_size)


        for batch_idx in range(num_batches):

            obs, _, _, _, _ = self.memory.sample_buffer(1)

            obs = self.normalize_observation(obs)
            pred_action = None

            for step_idx in range(batch_size):

                idx = batch_idx * batch_size + step_idx
                # Select action using world model action prediction with epsilon-greedy
                with torch.no_grad():
                    if pred_action is None or random.random() < self.epsilon:
                        # First step or explore: random action
                        action_idx = torch.tensor([random.randint(0, self.env.action_space.n - 1)], device=obs.device)
                    else:
                        # Exploit: use world model's action prediction from previous step
                        action_idx = pred_action.argmax(dim=1)
                    action_onehot = F.one_hot(action_idx, num_classes=self.env.action_space.n).float()

                next_obs, reward, pred_action, done = self.world_model(obs, action_onehot)

                states[idx]      = obs.squeeze(0)
                actions[idx]     = action_idx.item()
                rewards[idx]     = reward.item()
                next_states[idx] = next_obs.squeeze(0)
                dones[idx]       = (torch.sigmoid(done) > 0.5).float().item()

                obs = next_obs

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        return states, actions, rewards, next_states, dones

    def train_world_model(self, epochs, batch_size):

        total_reward_loss = 0
        total_action_loss = 0
        total_done_loss = 0
        total_next_frame_loss = 0
        total_combined_loss = 0
        
        for i in range(epochs):

            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(batch_size)

            next_obs_normalized = self.normalize_observation(next_obs)
            obs_normalized = self.normalize_observation(obs)

            # One-hot encode actions for world model input
            actions_onehot = F.one_hot(actions.long(), num_classes=self.env.action_space.n).float()

            pred_next_frame, pred_rewards, pred_actions, pred_dones = self.world_model.forward(obs_normalized, actions_onehot)

            reward_loss = F.mse_loss(pred_rewards.squeeze(-1), rewards)
            action_loss = F.cross_entropy(pred_actions, actions.long())
            done_loss = F.binary_cross_entropy_with_logits(pred_dones.squeeze(-1), dones.float())

            # next_frame_loss = F.l1_loss(pred_next_frame, next_obs_normalized)
            next_frame_loss = self.next_frame_loss(pred_next_frame, next_obs_normalized)

            combined_loss = reward_loss + action_loss + done_loss + next_frame_loss

            self.world_model_optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()

            # Just for stats.
            total_reward_loss += reward_loss.item()
            total_action_loss += action_loss.item()
            total_done_loss += done_loss.item()
            total_next_frame_loss += next_frame_loss.item()
            total_combined_loss += combined_loss.item()

        avg_combined_loss    = total_combined_loss / epochs
        avg_reward_loss      = total_reward_loss / epochs
        avg_action_loss      = total_action_loss / epochs
        avg_done_loss        = total_done_loss / epochs
        avg_next_frame_loss  = total_next_frame_loss / epochs

        return avg_combined_loss, avg_reward_loss, avg_action_loss, avg_done_loss, avg_next_frame_loss
        

            # Training
            # logits = self.reward_head(x)          # [batch, 1]
            # loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), reward.float())
            #
            # # Inference - hard
            # pred_reward = (logits.sigmoid() > 0.5).long().squeeze(-1)  # {0, 1}
            #
            # # Inference - soft (preferred for world model rollouts)
            # expected_reward = logits.sigmoid().squeeze(-1)  # continuous (0, 1)
            # pass

            # self.world_model.forward() 

    def train_q_model_step_live(self, batch_size):

        if self.memory.can_sample(batch_size):

            observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

            actions = actions.unsqueeze(1).long()
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1).float()

            q_values = self.q_model(observations)
            q_sa     = q_values.gather(1, actions)

            with torch.no_grad():
                next_actions = torch.argmax(
                    self.q_model(next_observations), dim=1, keepdim=True
                )

                next_q = self.target_q_model(next_observations).gather(1, next_actions)
                targets = rewards + (1 - dones) * self.gamma * next_q

            loss = F.mse_loss(q_sa, targets)

            self.q_model_optimizer.zero_grad()
            loss.backward()
            self.q_model_optimizer.step()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

            self.total_steps += 1

        return loss.item()


    def train_q_model_on_imagination(self, batch_size, num_batches, epochs=100):

        total_loss = 0
        
        for epoch in range(epochs):

            observations, actions, rewards, next_observations, dones = self.imagine_trajectory(batch_size, num_batches)

            actions = actions.unsqueeze(1).long()
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1).float()

            q_values = self.q_model(observations)
            q_sa     = q_values.gather(1, actions)

            with torch.no_grad():
                next_actions = torch.argmax(
                    self.q_model(next_observations), dim=1, keepdim=True
                )

                next_q = self.target_q_model(next_observations).gather(1, next_actions)
                targets = rewards + (1 - dones) * self.gamma * next_q

            loss = F.mse_loss(q_sa, targets)

            self.q_model_optimizer.zero_grad()
            loss.backward()
            self.q_model_optimizer.step()

            if self.total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

            total_loss += loss.item()

            self.total_steps += 1

        return total_loss / epochs



    def evaluate_rollout(self, num_steps=8, filename="eval_rollout.png"):
        """Evaluate world model rollout quality over multiple steps.

        Args:
            num_steps: Number of rollout steps to visualize
            filename: Output image path
        """
        if not self.memory.can_sample(1):
            return

        # Sample a starting observation
        obs, _, _, _, _ = self.memory.sample_buffer(1)
        obs = self.normalize_observation(obs)  # [1, 4, 96, 96]

        rollout_frames = [("step_0_real", obs.cpu())]
        current_obs = obs

        with torch.no_grad():
            for step in range(1, num_steps + 1):
                # Use world model to predict action from current observation
                _, _, action_logits, _ = self.world_model.forward(current_obs, None)
                action_pred = torch.argmax(action_logits, dim=1)  # [1]

                # Create one-hot action
                action_onehot = F.one_hot(action_pred, num_classes=self.env.action_space.n).float()

                # Predict next observation (full 4-frame stack)
                pred_next_obs, _, _, _ = self.world_model.forward(current_obs, action_onehot)

                # Store the predicted observation
                rollout_frames.append((f"step_{step}_pred", pred_next_obs.cpu()))

                # Use predicted observation as input for next step
                current_obs = pred_next_obs

        # Visualize all rollout steps (show only last frame of each 4-frame stack)
        display_stacked_obs(rollout_frames, filename, num_frames=1)

    def save(self):
        self.world_model.save_the_model("world_model", verbose=True)
        self.q_model.save_the_model("q_model", verbose=True)

    def load(self):
        self.world_model.load_the_model("world_model", device=self.device)
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_the_model("q_model", device=self.device)

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs = self.process_observation(obs, clear_stack=True)
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    obs_t = obs.unsqueeze(0).float().to(self.device)
                    action = self.q_model(obs_t).argmax(dim=1).item()

                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                done = term or trunc
                episode_reward += reward
                obs = next_obs

            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")

        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards

    def train(self, episodes=1, offline_training_epochs=1, summary_writer_suffix="_wm", batch_size=1, num_batches=1, wm_batch_size=1, use_world_model=False):

        if use_world_model:
            run_tag = f'world_model_ote{offline_training_epochs}_dynamic_ratio_bs{batch_size}_wmbs_{wm_batch_size}'
        else:
            run_tag = f'live_bs{batch_size}'
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}'

        writer = SummaryWriter(summary_writer_name)

        for episode in range(episodes):
            
            obs, info = self.env.reset()

            obs = self.process_observation(obs, clear_stack=True)

            done = False
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0

            while not done:

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_t = obs.unsqueeze(0).float().to(self.device)
                        action = self.q_model(obs_t).argmax(dim=1).item()

                next_obs, reward, term, trunc, info = self.env.step(action)

                next_obs = self.process_observation(next_obs)

                done = (term or trunc)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                episode_reward += reward
                episode_steps += 1

                # Do live training, if we're not using the world model.
                if self.memory.can_sample(batch_size) and not use_world_model:
                        episode_loss += self.train_q_model_step_live(batch_size)

                obs = next_obs

                if(random.random() < 0.01):
                    with torch.no_grad():
                        obs_for_logging = obs.unsqueeze(dim=0).to(self.device)
                        obs_for_logging = self.normalize_observation(obs_for_logging)
                        action_onehot = F.one_hot(torch.tensor([action], device=self.device), num_classes=self.env.action_space.n).float()
                        pred_next_frame, _, _, _ = self.world_model.forward(obs_for_logging, action_onehot)
                        display_stacked_obs([
                            ("predicted", pred_next_frame.cpu().detach()),
                            ("actual",    obs_for_logging.cpu().detach()),
                        ], "next_frame_pred.png")


            # Adjust epsilon. 
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Log stats for the current training iteration 
            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            if use_world_model:
                # Interleaved training with dynamic wm_q_ratio
                current_ratio = get_wm_q_ratio(episode)

                total_combined_loss = 0
                total_reward_loss = 0
                total_action_loss = 0
                total_done_loss = 0
                total_next_frame_loss = 0
                total_q_loss = 0
                wm_updates = 0
                q_updates = 0

                for offline_epoch in range(offline_training_epochs):
                    # World model updates
                    for _ in range(current_ratio[0]):
                        combined_loss, reward_loss, action_loss, done_loss, next_frame_loss = self.train_world_model(epochs=1, batch_size=wm_batch_size)
                        total_combined_loss += combined_loss
                        total_reward_loss += reward_loss
                        total_action_loss += action_loss
                        total_done_loss += done_loss
                        total_next_frame_loss += next_frame_loss
                        wm_updates += 1

                    # Q-model updates (ratio[1]=0 means no Q training)
                    for _ in range(current_ratio[1]):
                        q_loss = self.train_q_model_on_imagination(batch_size, num_batches=num_batches, epochs=1)
                        total_q_loss += q_loss
                        q_updates += 1

                # Average the losses
                avg_combined_loss = total_combined_loss / wm_updates if wm_updates > 0 else 0
                avg_reward_loss = total_reward_loss / wm_updates if wm_updates > 0 else 0
                avg_action_loss = total_action_loss / wm_updates if wm_updates > 0 else 0
                avg_done_loss = total_done_loss / wm_updates if wm_updates > 0 else 0
                avg_next_frame_loss = total_next_frame_loss / wm_updates if wm_updates > 0 else 0
                episode_loss = total_q_loss / q_updates if q_updates > 0 else 0

                writer.add_scalar("World Model/combined_loss", avg_combined_loss, episode)
                writer.add_scalar("World Model/reward_loss", avg_reward_loss, episode)
                writer.add_scalar("World Model/action_loss", avg_action_loss, episode)
                writer.add_scalar("World Model/done_loss", avg_done_loss, episode)
                writer.add_scalar("World Model/next_frame_loss", avg_next_frame_loss, episode)
                writer.add_scalar("Train/wm_updates_per_episode", wm_updates, episode)
                writer.add_scalar("Train/q_updates_per_episode", q_updates, episode)
                writer.add_scalar("Train/updates_per_cycle_wm", current_ratio[0], episode)
                writer.add_scalar("Train/updates_per_cycle_q", current_ratio[1], episode)

                if episode % 100 == 0:
                    print(f"Completed episode {episode} - Reward loss: {avg_reward_loss}")

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon", self.epsilon, episode)

            if episode_steps > 0:
                # If we're doing live training, we need to divide by episode steps
                episode_loss = episode_loss if use_world_model else episode_loss / episode_steps
                writer.add_scalar("Train/avg_q_loss", episode_loss, episode)

            # Evaluate rollout quality once per episode
            if use_world_model:
                self.evaluate_rollout(num_steps=8, filename="eval_rollout.png")

            if episode % 10 == 0:
                self.save()




