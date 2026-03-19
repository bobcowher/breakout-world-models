import gymnasium as gym
from collections import deque
import time
import torch
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

        self.max_episode_steps = 500

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        # print(torch.squeeze(obs).shape)

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024).to(self.device)
        
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


    def imagine_trajectory(self, obs, batch_size):
        states      = np.zeros(batch_size)
        actions     = np.zeros(batch_size)
        rewards     = np.zeros(batch_size)
        next_states = np.zeros(batch_size)
        dones       = np.zeros(batch_size)

        for i in range(batch_size):
            next_obs, reward, action, done = self.world_model(obs)

            action = action.argmax(dim=1).item()

            states[i] = obs
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_obs
            dones = [i] = done
        
        

        return states, actions, rewards, next_states, dones

    def train_world_model(self, epochs):

        total_reward_loss = 0
        total_next_frame_loss = 0
        total_combined_loss = 0
        
        for i in range(epochs):

            obs, actions, rewards, next_obs, dones = self.memory.sample_buffer(self.world_model_batch_size)

            next_obs_normalized = self.normalize_observation(next_obs) 
            obs_normalized = self.normalize_observation(obs)

            pred_next_frame, pred_rewards = self.world_model.forward(obs_normalized)

            reward_loss = F.binary_cross_entropy_with_logits(pred_rewards.squeeze(-1), rewards)

            # next_frame_loss = F.l1_loss(pred_next_frame, next_obs_normalized)
            next_frame_loss = self.next_frame_loss(pred_next_frame, next_obs_normalized)

            combined_loss = reward_loss + next_frame_loss

            self.world_model_optimizer.zero_grad()
            combined_loss.backward()
            self.world_model_optimizer.step()

            # Just for stats.
            total_reward_loss += reward_loss.item()
            total_next_frame_loss += next_frame_loss.item()
            total_combined_loss += combined_loss.item() 
        
        avg_reward_loss = total_reward_loss / epochs
        avg_next_frame_loss = total_next_frame_loss / epochs
        avg_combined_loss = total_combined_loss / epochs

        return avg_combined_loss, avg_reward_loss, avg_next_frame_loss
        

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

    def train_q_model_live(self, batch_size, total_steps):

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

            if total_steps % self.target_update_interval == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())

        return loss.item()


    def train_q_model_on_imagination(self, batch_size, total_steps):
        
        observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)


        #
        #     actions = actions.unsqueeze(1).long()
        #     rewards = rewards.unsqueeze(1)
        #     dones = dones.unsqueeze(1).float()
        #
        #     q_values = self.q_model(observations)
        #     q_sa     = q_values.gather(1, actions)
        #
        #     with torch.no_grad():
        #         next_actions = torch.argmax(
        #             self.q_model(next_observations), dim=1, keepdim=True
        #         )
        #
        #         next_q = self.target_q_model(next_observations).gather(1, next_actions)
        #         targets = rewards + (1 - dones) * self.gamma * next_q
        #
        #     loss = F.mse_loss(q_sa, targets)
        #
        #     self.q_model_optimizer.zero_grad()
        #     loss.backward()
        #     self.q_model_optimizer.step()
        #
        #     if total_steps % self.target_update_interval == 0:
        #         self.target_q_model.load_state_dict(self.q_model.state_dict())

        # return loss.item()

        return 0



    def train(self, episodes=1, world_model_epochs=1, summary_writer_suffix="_wm", batch_size=32, use_world_model=False):

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'

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

                self.total_steps += 1
                episode_reward += reward
                episode_steps += 1


                if self.memory.can_sample(batch_size):
                    if use_world_model:
                        episode_loss += self.train_q_model_on_imagination(batch_size, self.total_steps)
                    else:
                        episode_loss += self.train_q_model_live(batch_size, self.total_steps)
                

                obs = next_obs

                if(random.random() < 0.01):
                    with torch.no_grad():
                        obs_for_logging = obs.unsqueeze(dim=0).to(self.device)
                        obs_for_logging = self.normalize_observation(obs_for_logging)
                        pred_next_frame, _ = self.world_model.forward(obs_for_logging)
                        display_stacked_obs([
                            ("predicted", pred_next_frame.cpu().detach()),
                            ("actual",    obs_for_logging.cpu().detach()),
                        ], "next_frame_pred.png")

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon", self.epsilon, episode)

            if episode_steps > 0 and not use_world_model:
                writer.add_scalar("Train/avg_q_loss", episode_loss / episode_steps, episode)

            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")


            if use_world_model:
                combined_loss, reward_loss, next_frame_loss = self.train_world_model(epochs=world_model_epochs)

                writer.add_scalar("World Model/combined_loss", combined_loss, episode)
                writer.add_scalar("World Model/reward_loss", reward_loss, episode)
                writer.add_scalar("World Model/next_frame_loss", next_frame_loss, episode)

                if episode % 100 == 0:
                    print(f"Completed episode {episode} - Reward loss: {reward_loss}")




        self.memory.print_stats()


                # time.sleep(0.01)

                



