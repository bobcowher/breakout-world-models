import gymnasium as gym
from collections import deque
import time
import torch
from buffer import ReplayBuffer
from utils import display_stacked_obs
from models.world_model import WorldModel
import cv2

class Agent:

    def __init__(self, env : gym.Env,
                       max_buffer_size : int = 10000) -> None:
        self.env = env
        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.max_episode_steps = 500

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        # print(torch.squeeze(obs).shape)

        self.world_model = WorldModel(observation_shape=obs.shape, embed_dim=1024)
        
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)


    
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

        obs_stacked = torch.cat(tuple(self.frames), dim=0)

        return obs_stacked

    def train_world_model(self, epochs):
        
        for i in range(epochs):
            pass

            # self.world_model.forward() 


    def train(self, episodes=1):

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
                
                display_stacked_obs(obs)

                done = (term or trunc)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                print(reward)

                self.world_model(torch.unsqueeze(obs, dim=0))
                # self.world_model(obs)


        self.memory.print_stats()


                # time.sleep(0.01)

                



