"""cond_tokens='spherecloud' + pose_tokens='frame_pin': P1's GATr vector field fed the SO(3)-Fourier head's
sphere (88 hemisphere points + lifted signal) instead of the 7x7 cells; still exactly equivariant to 90 degree
turns and left-right flips of the backbone map."""

import math

import pytest
import torch

pytest.importorskip("healpy")

from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix  # noqa: E402

R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
FLIP = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))


def _flow(dtype=torch.float64, scalars=8, gatr=None):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="spherecloud", pose_tokens="frame_pin",
                         condition_head="none", dircloud_scalars=scalars, gatr_zero_readout=False,
                         gatr=gatr or dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).to(dtype)
    with torch.no_grad():
        model.adapter.conv_adapter.lift.radial_gain.normal_(1, 0.3)
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
    return model.eval()


def _velocity_from_map(model, fmap, R, t):
    return model.velocity(matrix_to_rotor(R), t, model.condition_head(model.adapter.conv_adapter(fmap)))


def _inputs(model, seed):
    torch.manual_seed(seed)
    return (torch.randn(2, 2048, 7, 7, dtype=torch.float64), rotor_to_matrix(random_rotor(2).double(), model.algebra),
            torch.tensor([0.3, 0.7], dtype=torch.float64))


def test_cloud_layout():
    model = _flow()
    cloud = model.adapter.conv_adapter(torch.randn(2, 2048, 7, 7, dtype=torch.float64))
    assert cloud.shape == (2, 89, 8 + 8 + 1) and model.n_cond_mv == 89
    torch.testing.assert_close(cloud[:, :88, 1:4].norm(dim=-1), torch.ones(2, 88, dtype=cloud.dtype))
    assert (cloud[:, :88, 3] > 0).all() and (cloud[:, :88, 8:16] >= 0).all()
    assert cloud[:, 88, 3].eq(1).all() and cloud[:, 88, 16].eq(1).all()


def test_velocity_invariant_to_turning_image_and_pose_together():
    model = _flow()
    fmap, R, t = _inputs(model, 0)
    v = _velocity_from_map(model, fmap, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity_from_map(model, torch.rot90(fmap, 1, dims=(2, 3)), R_Z_MINUS_90 @ R, t), v,
                               atol=1e-8, rtol=1e-6)


def test_velocity_flips_with_a_mirrored_image():
    model = _flow()
    fmap, R, t = _inputs(model, 1)
    v = _velocity_from_map(model, fmap, R, t)
    sign = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(_velocity_from_map(model, torch.flip(fmap, dims=(3,)), FLIP @ R @ FLIP, t), sign * v,
                               atol=1e-8, rtol=1e-6)


def test_spherecloud_flow_trains_and_samples():
    model = _flow(torch.float32).train()
    img = torch.randn(2, 3, 224, 224)
    loss = model.compute_loss(img, rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    assert model.adapter.conv_adapter.lift.proj.weight.grad.abs().max() > 0
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)
