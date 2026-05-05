import numpy as np

from app.services.landmark_normalizer import (
    FEATURE_VECTOR_SIZE,
    SINGLE_HAND_SIZE,
    normalize_feature_vector,
    normalize_sequence,
    normalize_single_hand,
)


def _fake_hand(offset=(0.5, 0.5, 0.0), scale=0.1):
    pts = np.random.RandomState(0).rand(21, 3) * scale
    pts += np.array(offset)
    return pts.flatten().astype(np.float32)


def test_translation_invariance():
    h1 = _fake_hand(offset=(0.5, 0.5, 0.0))
    h2 = _fake_hand(offset=(0.1, 0.9, 0.0))  # same shape, shifted
    n1 = normalize_single_hand(h1)
    n2 = normalize_single_hand(h2)
    assert np.allclose(n1, n2, atol=1e-6)


def test_scale_invariance():
    h1 = _fake_hand(scale=0.1)
    h2 = _fake_hand(scale=0.5)  # same hand bigger
    n1 = normalize_single_hand(h1)
    n2 = normalize_single_hand(h2)
    assert np.allclose(n1, n2, atol=1e-6)


def test_zero_hand_stays_zero():
    h = np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)
    out = normalize_single_hand(h)
    assert np.all(out == 0)


def test_feature_vector_handles_two_hands():
    left = _fake_hand()
    right = _fake_hand(offset=(0.2, 0.3, 0.0))
    vec = np.concatenate([left, right])
    assert vec.shape == (FEATURE_VECTOR_SIZE,)
    out = normalize_feature_vector(vec)
    assert out.shape == (FEATURE_VECTOR_SIZE,)


def test_sequence_normalization_preserves_shape():
    seq = np.random.rand(30, FEATURE_VECTOR_SIZE).astype(np.float32)
    out = normalize_sequence(seq)
    assert out.shape == (30, FEATURE_VECTOR_SIZE)
