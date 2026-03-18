from agent import Agent
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

agent = Agent(env=env, world_model_batch_size=32, max_buffer_size=15000) 

agent.train(episodes=5000, world_model_epochs=100)
