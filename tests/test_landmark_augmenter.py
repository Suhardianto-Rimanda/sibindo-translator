import numpy as np

from app.services.landmark_augmenter import augment, jitter, rotate, scale, temporal_shift
from app.services.landmark_normalizer import FEATURE_VECTOR_SIZE


def _fake_seq():
    np.random.seed(42)
    return np.random.rand(30, FEATURE_VECTOR_SIZE).astype(np.float32)


def test_rotate_preserves_shape():
    seq = _fake_seq()
    out = rotate(seq, max_deg=20)
    assert out.shape == seq.shape


def test_scale_preserves_shape():
    seq = _fake_seq()
    out = scale(seq, low=0.8, high=1.2)
    assert out.shape == seq.shape


def test_jitter_only_perturbs_nonzero():
    seq = np.zeros((10, FEATURE_VECTOR_SIZE), dtype=np.float32)
    seq[:, 0] = 0.5
    out = jitter(seq, sigma=0.1)
    # zero entries stay zero
    assert np.all(out[:, 1:] == 0)
    # non-zero entries got perturbed
    assert not np.allclose(out[:, 0], seq[:, 0])


def test_temporal_shift_returns_same_length():
    seq = _fake_seq()
    out = temporal_shift(seq, max_shift=5)
    assert out.shape == seq.shape


def test_augment_combined():
    seq = _fake_seq()
    out = augment(seq, p=1.0)
    assert out.shape == seq.shape
    assert not np.allclose(out, seq)
