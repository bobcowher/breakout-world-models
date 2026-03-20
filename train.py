from agent import Agent
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

agent = Agent(env=env, world_model_batch_size=32, max_buffer_size=20000) 

agent.train(episodes=5000, q_model_epochs=200, world_model_epochs=100, use_world_model=True, batch_size=8, num_batches=8)
