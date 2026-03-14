import gymnasium as gym
from collections import deque
import time
import torch
from buffer import ReplayBuffer
from utils import display_stacked_obs
from models.world_model import WorldModel
import cv2
import torch.nn.functional as F

class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000,
                       world_model_batch_size = 8) -> None:
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
        
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)

        self.world_model_batch_size = world_model_batch_size


    
    def init_frame_stack(self, obs):
        """Call once after env.reset().  Pre-fill both deques."""
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)

    def process_observation(self, obs, clear_stack=False):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)  

        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_NEAREST) # shrink to 128
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

    def train_world_model(self, epochs):

        total_reward_loss = 0
        
        for i in range(epochs):

            states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.world_model_batch_size)

            pred_rewards = self.world_model.forward(states)

            reward_loss = F.binary_cross_entropy_with_logits(rewards, pred_rewards.squeeze(-1))

            self.world_model_optimizer.zero_grad()
            reward_loss.backward()
            self.world_model_optimizer.step()

            total_reward_loss += reward_loss.item()

        
        avg_reward_loss = total_reward_loss / epochs

        return avg_reward_loss
        

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


    def train(self, episodes=1, world_model_epochs=1):

        total_steps = 0
         
        for episode in range(episodes):
            
            obs, info = self.env.reset()

            obs = self.process_observation(obs, clear_stack=True)

            done = False
            episode_step = 0

            while not done:

                action = self.env.action_space.sample()

                next_obs, reward, term, trunc, info = self.env.step(action)

                next_obs = self.process_observation(next_obs)
                
                # display_stacked_obs(obs)

                done = (term or trunc)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                # self.world_model(torch.unsqueeze(obs, dim=0))

            reward_loss = self.train_world_model(epochs=world_model_epochs)

            if episode % 100 == 0:
                print(f"Completed episode {episode} - Reward loss: {reward_loss}")
                # self.world_model(obs)


        self.memory.print_stats()


                # time.sleep(0.01)

                



