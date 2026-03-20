import gymnasium as gym


class LifePenaltyWrapper(gym.Wrapper):
    """Wrapper that applies a penalty when the agent loses a life in Atari games.

    Args:
        env: The environment to wrap
        penalty: The reward penalty to apply when a life is lost (default: -1.0)
    """

    def __init__(self, env, penalty=-1.0):
        super().__init__(env)
        self.penalty = penalty
        self.lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        current_lives = info.get('lives', 0)

        # Apply penalty if lives decreased
        if current_lives < self.lives:
            reward += self.penalty

        self.lives = current_lives
        return obs, reward, term, trunc, info
