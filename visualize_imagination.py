import sys
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # prevent Qt thread-affinity warnings from ALE threads
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ale_py
import gymnasium as gym
from life_penalty_wrapper import LifePenaltyWrapper
from agent import Agent


SCALE     = 5   # display scale factor (96 → 480)
FPS       = 1   # playback speed (1 second per frame)
GAP       = 4   # pixel gap between real and imagined panels
MAX_STEPS = 32  # extended rollout for reward correlation analysis


def hidden_state_to_frame(hidden_state, world_model):
    """Decode an RSSM hidden state to a uint8 HWC numpy frame."""
    with torch.no_grad():
        img = world_model.decode(hidden_state)  # (1, C, H, W) in [0,1]
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def obs_to_frame(obs_tensor):
    """Convert a (C, H, W) uint8 tensor to HWC numpy."""
    return obs_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def annotate(frame, label, reward=None):
    frame = frame.copy()
    cv2.putText(frame, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if reward is not None:
        r_str = f"r={reward:+.2f}"
        cv2.putText(frame, r_str, (4, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 255, 100), 1, cv2.LINE_AA)
    return frame


def scale_up(frame, factor):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def make_display(real_frame, imag_frame, step, real_reward, imag_reward):
    real_ann  = annotate(real_frame,  f"Real  step {step}", reward=real_reward)
    imag_ann  = annotate(imag_frame,  f"Imag  step {step}", reward=imag_reward)
    real_big  = scale_up(real_ann, SCALE)
    imag_big  = scale_up(imag_ann, SCALE)
    gap       = np.zeros((real_big.shape[0], GAP, 3), dtype=np.uint8)
    combined  = np.concatenate([real_big, gap, imag_big], axis=1)
    # OpenCV expects BGR
    return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = LifePenaltyWrapper(env, penalty=-1.0)

    agent = Agent(env=env)

    for path in ["checkpoints/world_model.pt", "checkpoints/q_model.pt"]:
        if not os.path.exists(path):
            print(f"ERROR: checkpoint not found: {path}")
            env.close()
            sys.exit(1)

    agent.load()
    agent.world_model.eval()
    agent.q_model.eval()

    n_actions = agent.world_model.n_actions

    def get_initial_hidden_state():
        """Warm up with 30 random steps, then encode the final obs into an RSSM hidden state."""
        obs, _ = env.reset()
        for i in range(30):
            action = 1 if i == 0 else env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
        obs_tensor   = agent.process_observation(obs)
        obs_norm     = obs_tensor.float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            hidden_state = agent.world_model.encode_observation_to_hidden(obs_norm)
        return obs_tensor, hidden_state

    cv2.namedWindow("Imagination Rollout", cv2.WINDOW_AUTOSIZE)
    delay_ms = int(1000 / FPS)

    obs_tensor, current_hidden_state = get_initial_hidden_state()
    # Show the real starting frame on both panels before the loop begins
    start_frame = obs_to_frame(obs_tensor)
    cv2.imshow("Imagination Rollout", make_display(start_frame, start_frame, 0, 0.0, 0.0))
    cv2.waitKey(delay_ms)

    step = 0
    print("Running imagination rollout. Press 'q' to quit.")

    try:
        while True:
            with torch.no_grad():
                # === Imagination: select action from RSSM hidden state (closed loop) ===
                action_idx    = agent.q_model(current_hidden_state).argmax(dim=1)  # (1,)
                action_onehot = F.one_hot(action_idx, num_classes=n_actions).float()

                next_hidden_state, reward_pred, done_pred = agent.world_model.imagine_step(
                    current_hidden_state, action_onehot
                )
                imag_reward = reward_pred.item()
                imag_done   = done_pred.item() > 0.5

                # === Real env: step with same action ===
                next_obs_raw, real_reward, real_term, real_trunc, _ = env.step(action_idx.item())
                next_obs_tensor = agent.process_observation(next_obs_raw)

                # Decode imagined frame from RSSM hidden state
                imag_frame = hidden_state_to_frame(next_hidden_state, agent.world_model)
                real_frame = obs_to_frame(next_obs_tensor)

            panel = make_display(real_frame, imag_frame, step, real_reward, imag_reward)
            cv2.imshow("Imagination Rollout", panel)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                break

            current_hidden_state = next_hidden_state
            step += 1

            if imag_done or real_term or real_trunc or step >= MAX_STEPS:  # noqa
                print(f"Episode ended at step {step}. Closing in 10 seconds (press 'q' to quit early).")
                if cv2.waitKey(10000) & 0xFF == ord('q'):
                    break
                break

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
