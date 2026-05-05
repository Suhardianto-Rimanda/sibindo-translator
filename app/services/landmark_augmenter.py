"""Data augmentation for hand landmark sequences.

Operates on shape (frames, 126). Applies:
- Random rotation (around wrist, 2D in XY plane)
- Random scale jitter
- Gaussian noise injection
- Random temporal shift (drop/duplicate frames)
"""
import numpy as np

from app.services.landmark_normalizer import (
    NUM_HAND_LANDMARKS,
    LANDMARK_DIMS,
    SINGLE_HAND_SIZE,
)


def _rotate_hand(coords: np.ndarray, angle_rad: float) -> np.ndarray:
    if np.all(coords == 0):
        return coords
    pts = coords.reshape(NUM_HAND_LANDMARKS, LANDMARK_DIMS).copy()
    wrist = pts[0].copy()
    pts -= wrist
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ rot.T
    pts += wrist
    return pts.flatten().astype(np.float32)


def rotate(seq: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    out = np.empty_like(seq)
    for i, frame in enumerate(seq):
        left = _rotate_hand(frame[:SINGLE_HAND_SIZE], angle)
        right = _rotate_hand(frame[SINGLE_HAND_SIZE:], angle)
        out[i] = np.concatenate([left, right])
    return out


def scale(seq: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    factor = np.random.uniform(low, high)
    return (seq * factor).astype(np.float32)


def jitter(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    mask = (seq != 0).astype(np.float32)
    return (seq + noise * mask).astype(np.float32)


def temporal_shift(seq: np.ndarray, max_shift: int = 3) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq.copy()
    if shift > 0:
        return np.concatenate([np.tile(seq[0:1], (shift, 1)), seq[:-shift]], axis=0)
    return np.concatenate([seq[-shift:], np.tile(seq[-1:], (-shift, 1))], axis=0)


def augment(seq: np.ndarray, p: float = 0.7) -> np.ndarray:
    """Apply random augmentations with probability p each."""
    out = seq.copy()
    if np.random.rand() < p:
        out = rotate(out)
    if np.random.rand() < p:
        out = scale(out)
    if np.random.rand() < p:
        out = jitter(out)
    if np.random.rand() < p:
        out = temporal_shift(out)
    return out
