"""Normalize hand landmarks to be invariant to position, scale, and camera distance.

Two normalization strategies:
- Wrist-relative: subtract wrist (landmark 0) from all points -> translation invariant.
- Bbox-scaled: divide by max distance from wrist -> scale invariant.

Combined: position + scale invariant. Z-axis preserved as relative depth.
"""
import numpy as np


NUM_HAND_LANDMARKS = 21
LANDMARK_DIMS = 3
SINGLE_HAND_SIZE = NUM_HAND_LANDMARKS * LANDMARK_DIMS  # 63
FEATURE_VECTOR_SIZE = SINGLE_HAND_SIZE * 2             # 126 (left + right)


def normalize_single_hand(coords: np.ndarray) -> np.ndarray:
    """Normalize one hand's flat 63-dim vector. Returns same shape.

    coords layout: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """
    if np.all(coords == 0):
        return coords

    pts = coords.reshape(NUM_HAND_LANDMARKS, LANDMARK_DIMS).copy()
    wrist = pts[0].copy()
    pts -= wrist  # translation invariance

    # scale invariance: divide by max abs distance from wrist (excluding wrist)
    max_dist = np.max(np.linalg.norm(pts[1:], axis=1))
    if max_dist > 1e-6:
        pts /= max_dist

    return pts.flatten().astype(np.float32)


def normalize_feature_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize 126-dim vector (two hands) independently."""
    left = normalize_single_hand(vec[:SINGLE_HAND_SIZE])
    right = normalize_single_hand(vec[SINGLE_HAND_SIZE:])
    return np.concatenate([left, right])


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize sequence of shape (frames, 126)."""
    return np.array([normalize_feature_vector(f) for f in seq], dtype=np.float32)
