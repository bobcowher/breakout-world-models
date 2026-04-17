from agent import Agent
import gymnasium as gym
import ale_py
from life_penalty_wrapper import LifePenaltyWrapper

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = LifePenaltyWrapper(env, penalty=-1.0)

agent = Agent(env=env, max_buffer_size=100000) 

agent.train(episodes=1200, offline_training_epochs=200, batch_size=16, wm_batch_size=32, num_batches=8)
