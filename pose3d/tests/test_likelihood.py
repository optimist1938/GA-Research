"""engine/likelihood.py against a flow with a closed-form density.

v = -alpha log(R) shrinks every rotation angle, theta_1 = theta_0 exp(-alpha), about a fixed axis. The Haar
measure is (1 - cos theta) dtheta dOmega / pi^2 (volume pi^2), so from the uniform start
p_1(R) = exp(alpha) (1 - cos(theta exp(alpha))) / (1 - cos theta) / pi^2 for theta < pi exp(-alpha).
"""

import math

import torch
import torch.nn as nn
from clifford.algebra.cliffordalgebra import CliffordAlgebra

import pose3d.geometry.matrix as matrix_flow
from pose3d.engine.likelihood import (
    LOG_UNIFORM, _MatrixStates, _RotorStates, flow_log_likelihood, symsol_log_likelihood)
from pose3d.geometry.flow import log_map as rotor_log
from pose3d.geometry.rotor import matrix_to_rotor

ALPHA = 0.5


class _ShrinkMatrix(nn.Module):
    def velocity(self, R, t, cond):
        return -ALPHA * matrix_flow.log_map(R)


class _ShrinkRotor(nn.Module):
    algebra = CliffordAlgebra((1, 1, 1))

    def velocity(self, r, t, cond):
        return -ALPHA * rotor_log(r)


def _poses(n, lo=0.3, hi=1.6):
    torch.manual_seed(0)
    axis = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    theta = torch.linspace(lo, hi, n)
    return matrix_flow.exp_map(axis * theta[:, None]), theta


def _exact(theta):
    return (ALPHA + torch.log(1 - torch.cos(theta * math.exp(ALPHA))) - torch.log(1 - torch.cos(theta))
            + LOG_UNIFORM)


def test_matrix_flow_matches_the_closed_form():
    R, theta = _poses(64)
    model = _ShrinkMatrix()
    ll = flow_log_likelihood(model, None, R, steps=50, states=_MatrixStates(model))
    torch.testing.assert_close(ll, _exact(theta), atol=5e-3, rtol=0)


def test_rotor_flow_matches_the_closed_form():
    R, theta = _poses(64)
    model = _ShrinkRotor()
    ll = flow_log_likelihood(model, None, R, steps=50, states=_RotorStates(model))
    torch.testing.assert_close(ll, _exact(theta), atol=5e-3, rtol=0)


def test_zero_field_is_uniform():
    class Zero(nn.Module):
        def velocity(self, R, t, cond):
            return 0.0 * R[:, 0]

    R, _ = _poses(8)
    ll = flow_log_likelihood(Zero(), None, R, steps=4, states=_MatrixStates(None))
    torch.testing.assert_close(ll, torch.full((8,), LOG_UNIFORM))


def test_symsol_average_over_equivalents():
    """With fewer equivalents than requested, each pose counts once (repeats carry no weight)."""

    class Flow(_ShrinkMatrix):
        def condition(self, img):
            return torch.zeros(img.shape[0], 1)

    R, theta = _poses(5)
    rots = torch.zeros(2, 360, 3, 3)
    rots[0, :3], rots[1, :5] = R[:3], R
    batch = dict(img=torch.zeros(2, 3, 4, 4), rots_eq=rots, n_eq=torch.tensor([3, 5]),
                 cls_eval=torch.tensor([[0], [1]]))
    ll, cls = symsol_log_likelihood(Flow(), [batch], "cpu", n_equivalents=4, steps=50)
    exact = _exact(theta)
    assert cls.tolist() == [0, 1]
    assert abs(ll[0] - exact[:3].mean().item()) < 5e-3
    assert ll[1] < exact.max().item() and ll[1] > exact.min().item()
