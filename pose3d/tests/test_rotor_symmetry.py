import numpy as np
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.analyze_rotor_symmetry import analyze, rot_err, summarize


class _Stub(torch.nn.Module):
    """velocity = a fixed linear map of the rotor's bivector part: odd in the rotor (v(-r) = -v(r)) unless `even`, then of its square."""

    def __init__(self, even):
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1)); self.even = even; self.w = torch.nn.Parameter(torch.zeros(1))

    def condition(self, img):
        return torch.zeros(1, 1, 8)

    def velocity(self, rotor, t, cond):
        b = rotor[:, 1:]
        return b * rotor[:, :1] if self.even else b                 # (w, b) -> w*b is even; b alone is odd


def test_rot_err_treats_q_and_minus_q_as_one_rotation():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rot_err(np.stack([q, -q]), q), 0)
    half = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0])         # a 45 degree rotation
    assert abs(rot_err(half, q) - 45) < 1e-6 and abs(rot_err(-half, q) - 45) < 1e-6


def test_evenness_measure_separates_even_and_odd_fields():
    imgs = torch.zeros(2, 3, 8, 8); rots = np.stack([np.eye(3)] * 2)
    for even, expect in ((True, 0.0), (False, 2.0)):
        rows, mirror, split = analyze(_Stub(even), imgs, rots, [0, 1], k=6, steps=4, log=lambda s: None)
        res = summarize(rows, mirror, split)
        assert abs(res["evenness_all_trajectories_median"] - expect) < 1e-5
        assert res["split"]["to_plus_q"] + res["split"]["to_minus_q"] + res["split"]["miss"] == 12
