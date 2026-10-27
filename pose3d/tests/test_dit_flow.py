"""DiTFlow: shapes, identity start, gradients, sampling, size and registry (no pretrained weights)."""

import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.config import parse_args
from pose3d.models import build_model
from pose3d.models.dit_flow import DiTFlow

ALGEBRA = CliffordAlgebra((1, 1, 1))


def _flow(**kwargs):
    return DiTFlow(ALGEBRA, pretrained_backbone=False, encoder_type="resnet50", n_time_samples=2, **kwargs)


def _n(module):
    return sum(p.numel() for p in module.parameters())


def test_velocity_shape_and_identity_start():
    flow = _flow(dim=32, blocks=2, heads=2).eval()
    img = torch.rand(2, 3, 64, 64)
    cond = flow.condition(img)
    assert cond.shape == (2, 64, 32)
    R = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    v = flow.velocity(R, torch.rand(2), cond)
    assert v.shape == (2, 3)
    assert torch.count_nonzero(v) == 0   # adaLN-Zero + zero output layer: the flow starts as the identity


def test_loss_backward_reaches_the_denoiser():
    flow = _flow(dim=32, blocks=2, heads=2)
    loss = flow.compute_loss(torch.rand(3, 3, 64, 64), torch.eye(3).expand(3, 3, 3).clone(), cls=None)
    loss.backward()
    assert torch.isfinite(loss)
    assert flow.vector_field.out.weight.grad.abs().sum() > 0


def test_predict_returns_rotations():
    flow = _flow(dim=32, blocks=2, heads=2).eval()
    for n_samples in (1, 4):
        R = flow.predict(torch.rand(2, 3, 64, 64), None, n_samples=n_samples, steps=3)
        assert R.shape == (2, 3, 3)
        torch.testing.assert_close(R @ R.transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-4, rtol=0)


def test_default_size_matches_cliffordflow_budget():
    flow = _flow()
    heads = _n(flow.condition_head) + _n(flow.vector_field)
    assert heads == 1_189_203, heads   # CliffordFlow's two heads: 1,133,689


def test_registry_and_flags():
    cfg = parse_args(["--path_to_datasets=x", "--model", "dit_flow", "--no-pretrained_backbone",
                      "--encoder", "resnet50", "--dit_dim", "48", "--dit_blocks", "2"])
    model = build_model(cfg, ALGEBRA)
    assert isinstance(model, DiTFlow)
    assert len(model.vector_field.blocks) == 2 and model.vector_field.out.in_features == 48
    assert model.n_time_samples == cfg.n_time_samples


def test_wigner_pose_features():
    """--dit_pose_lmax 6: the pose enters as D^l(R), l = 1..6 (454 numbers), e3nn's Wigner matrices; lmax 0 is the
    plain matrix. The features are smooth in R (no Euler angles), so the flow likelihood can differentiate them."""
    from e3nn import o3

    vf = _flow(dim=32, blocks=2, heads=2, pose_lmax=6).vector_field
    R = torch.linalg.qr(torch.randn(3, 3, 3))[0]
    R = R * torch.det(R).sign()[:, None, None]
    f = vf.pose_features(R)
    assert f.shape == (3, 454)
    ref = torch.cat([o3.wigner_D(l, *o3.matrix_to_angles(R)).flatten(1) for l in range(1, 7)], dim=1)
    torch.testing.assert_close(f, ref, atol=1e-4, rtol=1e-4)
    assert torch.equal(_flow(dim=32, blocks=2, heads=2).vector_field.pose_features(R), R.flatten(1))
    e = torch.zeros(3, 3, requires_grad=True)
    K = torch.zeros(3, 3, 3)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -e[:, 2], e[:, 1], -e[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = e[:, 2], -e[:, 1], e[:, 0]
    g = torch.autograd.grad(vf.pose_features(torch.eye(3).expand(3, 3, 3) @ (torch.eye(3) + K)).sum(), e)[0]
    assert torch.isfinite(g).all()                     # at the identity, where Euler angles are singular


@pytest.mark.parametrize("pose_lmax", [0, 6])
def test_dit_on_the_sphere_tokens(pose_lmax):
    pytest.importorskip("healpy")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.dit_flow import DiTFlow
    from pose3d.geometry.matrix import random_rotation_matrices

    torch.manual_seed(0)
    model = DiTFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50", dim=16, heads=2,
                    blocks=1, n_cond_tokens=8, pose_lmax=pose_lmax, sphere_input=True, sphere_channels=4)
    img = torch.randn(2, 3, 224, 224)
    tokens = model.adapter(img)
    assert tokens.shape == (2, 88, 4 + 3)
    torch.testing.assert_close(tokens[0, :, 4:], model.adapter.adapter.conv_adapter.points.float())
    loss = model.compute_loss(img, random_rotation_matrices(2))
    loss.backward()
    assert torch.isfinite(loss) and model.vector_field.out.weight.grad.abs().max() > 0   # adaLN-zero start
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)
