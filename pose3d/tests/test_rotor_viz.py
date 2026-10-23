import numpy as np
import pytest
import torch

from pose3d.engine.rotor_viz import destinations, gt_rotor, stereo_frame, stereographic
from pose3d.geometry.rotor import rotor_to_matrix
from scipy.spatial.transform import Rotation


def test_both_rotors_are_the_truth_and_land_on_plus_minus_one():
    from clifford.algebra.cliffordalgebra import CliffordAlgebra

    alg = CliffordAlgebra((1, 1, 1))
    R = Rotation.random(1, random_state=0).as_matrix()[0]
    q = gt_rotor(R)
    assert np.isclose(np.linalg.norm(q), 1) and q[0] >= 0
    mats = rotor_to_matrix(torch.from_numpy(np.stack([q, -q])).float(), alg).numpy()
    assert np.allclose(mats, R, atol=1e-4)                       # q and -q: the same rotation
    path = np.random.default_rng(0).normal(size=(5, 7, 4)); path /= np.linalg.norm(path, axis=-1, keepdims=True)
    frame = stereo_frame(q, path)
    e0, e1, e2, P = frame
    assert np.allclose([e0 @ e1, e0 @ e2, e0 @ P, e1 @ e2, e1 @ P, e2 @ P], 0, atol=1e-9) and np.isclose(P @ P, 1)
    assert np.allclose(stereographic(np.stack([q, -q]), frame), [[1, 0, 0], [-1, 0, 0]], atol=1e-9)


def test_stereographic_is_invertible_and_destinations_pick_the_nearer_rotor():
    q = np.array([0.8, 0.0, 0.6, 0.0])
    x = np.random.default_rng(1).normal(size=(50, 4)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    frame = stereo_frame(q, x[None])
    y = stereographic(x, frame)
    n2 = (y ** 2).sum(-1)                                          # inverse: x = (2y, n2 - 1)/(n2 + 1) in the (e0, e1, e2, P) basis
    back = np.einsum("nc,cd->nd", np.concatenate([2 * y, (n2 - 1)[:, None]], 1) / (n2 + 1)[:, None], np.stack(frame[:3] + (frame[3],)))
    assert np.allclose(back, x, atol=1e-8)
    off = np.array([0.0, 1.0, 0.0, 0.0])
    far = (q * 0.2 + off * 0.98); far /= np.linalg.norm(far)                      # far from both +-q
    near = np.stack([-q, q, far])
    sign, ok = destinations(near[None], q)                                        # path with a single step, three samples
    assert sign.tolist() == [-1, 1, 1] or sign[:2].tolist() == [-1, 1]
    assert ok.tolist() == [True, True, False]
