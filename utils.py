import matplotlib.pyplot as plt

plt.ion()

_fig = None
_axes = None
_imgs = None


def display_stacked_obs(obs, title="Stacked Observations", num_frames=4):
    """Display stacked observation as side-by-side frames.

    Args:
        obs: Observation array - frames stacked vertically (H*num_frames, W)
        title: Title for the figure
        num_frames: Number of frames stacked in the observation
    """
    global _fig, _axes, _imgs

    if obs.ndim == 3:
        if obs.shape[0] < obs.shape[-1]:
            obs = obs.transpose(1, 2, 0)
        if obs.shape[-1] in (1, 3, 4):
            obs = obs[:, :, 0]

    # Split vertically stacked frames
    h = obs.shape[0] // num_frames
    frames = [obs[i * h:(i + 1) * h, :] for i in range(num_frames)]

    if _fig is None or not plt.fignum_exists(_fig.number):
        _fig, _axes = plt.subplots(1, num_frames, figsize=(16, 6))
        _imgs = []
        for i, ax in enumerate(_axes):
            img = ax.imshow(frames[i], cmap='gray', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'Frame {i}')
            _imgs.append(img)
    else:
        for i, img in enumerate(_imgs):
            img.set_data(frames[i])

    _fig.suptitle(title)
    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()
