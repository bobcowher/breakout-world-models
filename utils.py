from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.image

plt.ion()

_fig: Optional[matplotlib.figure.Figure] = None
_imgs: Optional[list[matplotlib.image.AxesImage]] = None


def display_stacked_obs(obs, title="Stacked Observations", num_frames=4):
    """Display stacked observation as side-by-side frames.

    Args:
        obs: Observation array - (num_frames, H, W) or vertically stacked (H*num_frames, W)
        title: Title for the figure
        num_frames: Number of frames stacked in the observation
    """
    global _fig, _imgs

    if hasattr(obs, 'numpy'):
        obs = obs.numpy()

    if obs.ndim == 3 and obs.shape[0] == num_frames:
        # (num_frames, H, W) — frames stacked along dim 0
        frames = [obs[i] for i in range(num_frames)]
    else:
        if obs.ndim == 3:
            if obs.shape[0] < obs.shape[-1]:
                obs = obs.transpose(1, 2, 0)
            if obs.shape[-1] in (1, 3, 4):
                obs = obs[:, :, 0]
        # Split vertically stacked frames
        h = obs.shape[0] // num_frames
        frames = [obs[i * h:(i + 1) * h, :] for i in range(num_frames)]

    if _imgs is None:
        _fig, axes = plt.subplots(1, num_frames, figsize=(16, 6))
        _imgs = []
        for i, ax in enumerate(axes.flat):
            img = ax.imshow(frames[i], cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'Frame {i}')
            _imgs.append(img)
    else:
        for i, img in enumerate(_imgs):
            img.set_data(frames[i])

    assert _fig is not None
    _fig.suptitle(title)
    plt.pause(0.001)
