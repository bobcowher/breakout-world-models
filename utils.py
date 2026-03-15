from __future__ import annotations
import numpy as np
import cv2


def _to_frames(obs, num_frames: int) -> list[np.ndarray]:
    """Convert any supported obs shape to a list of (H, W) float32 frames."""
    if hasattr(obs, 'numpy'):
        obs = obs.numpy()

    obs = obs.astype(np.float32)

    if obs.ndim == 4:
        obs = obs[0]

    if obs.ndim == 3 and obs.shape[0] == num_frames:
        return [obs[i] for i in range(num_frames)]

    if obs.ndim == 3:
        if obs.shape[0] < obs.shape[-1]:
            obs = obs.transpose(1, 2, 0)
        if obs.shape[-1] in (1, 3, 4):
            obs = obs[:, :, 0]

    h = obs.shape[0] // num_frames
    return [obs[i * h:(i + 1) * h, :] for i in range(num_frames)]


def _labeled_row(label: str, frames: list[np.ndarray], label_height: int = 20) -> np.ndarray:
    """Tile frames side-by-side and prepend a label bar."""
    frames_u8 = [(np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) for f in frames]
    tiled = np.concatenate(frames_u8, axis=1)

    bar = np.zeros((label_height, tiled.shape[1]), dtype=np.uint8)
    cv2.putText(bar, label, (4, label_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)

    return np.concatenate([bar, tiled], axis=0)


def display_stacked_obs(entries: list[tuple[str, object]], filename: str, num_frames: int = 4):
    """Save one or more labeled stacked observations to an image file.

    Args:
        entries:   List of (label, obs) pairs. obs can be (B,C,H,W), (C,H,W), or (H*C,W).
        filename:  Output path (e.g. 'debug_pred.png')
        num_frames: Number of frames per stacked observation
    """
    rows = []
    for label, obs in entries:
        frames = _to_frames(obs, num_frames)
        rows.append(_labeled_row(label, frames))

    cv2.imwrite(filename, np.concatenate(rows, axis=0))
