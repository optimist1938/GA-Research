"""vector_field='rotorcorr' (GA-1): the read-out depends on (pose, image) only through <x_i, R p_j>, so it is
invariant to turning the image's points and the pose together by any rotation, and through the lift exactly
equivariant to 90 degree turns and left-right flips of the backbone map, like the SO(3)-Fourier head."""

import math

import pytest
import torch

pytest.importorskip("healpy")

from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix  # noqa: E402
from pose3d.models.rotor_corr_head import RotorCorrelationVelocity  # noqa: E402

R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
FLIP = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
S = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)


def _flow(dtype=torch.float64, flip_symmetrize=True):
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="rotorcorr", cond_tokens="sphere", condition_head="none",
                         rotorcorr=dict(c_s=8, templates=12, hidden=16, flip_symmetrize=flip_symmetrize)).to(dtype)
    with torch.no_grad():   # undo the zero-initialised read-out
        model.vector_field.out.weight.normal_(0, 0.5)
        model.vector_field.out.bias.normal_(0, 0.5)
        model.adapter.conv_adapter.radial_gain.normal_(1, 0.3)
    return model.eval()


def _velocity_from_map(model, fmap, R, t):
    return model.velocity(matrix_to_rotor(R), t, model.condition_head(model.adapter.conv_adapter(fmap)))


def _inputs(model, seed=0):
    torch.manual_seed(seed)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    return fmap, R, torch.tensor([0.3, 0.8], dtype=torch.float64)


def test_readout_invariant_under_any_rotation():
    torch.manual_seed(2)
    pts = torch.nn.functional.normalize(torch.randn(30, 3, dtype=torch.float64), dim=-1)
    vf = RotorCorrelationVelocity(pts, c_s=5, templates=7, hidden=9).double()
    with torch.no_grad():
        vf.out.weight.normal_(0, 0.5)
    q = torch.nn.functional.normalize(torch.randn(4, 4, dtype=torch.float64), dim=-1)
    from pose3d.engine.grid_likelihood import _quat_to_matrix
    R, G = _quat_to_matrix(q[:3]), _quat_to_matrix(q[3])
    f, t = torch.rand(3, 5, 30, dtype=torch.float64), torch.rand(3, dtype=torch.float64)
    v = vf(R, t, f)
    vf_turned = RotorCorrelationVelocity(pts @ G.T, c_s=5, templates=7, hidden=9).double()
    vf_turned.load_state_dict(vf.state_dict())
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(vf_turned(G @ R, t, f), v, atol=1e-10, rtol=1e-8)


def test_velocity_invariant_to_turning_image_and_pose_together():
    model = _flow()
    fmap, R, t = _inputs(model)
    v = _velocity_from_map(model, fmap, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity_from_map(model, torch.rot90(fmap, 1, dims=(2, 3)), R_Z_MINUS_90 @ R, t), v,
                               atol=1e-8, rtol=1e-6)


def test_velocity_flips_with_a_mirrored_image():
    model = _flow()
    fmap, R, t = _inputs(model, 1)
    v = _velocity_from_map(model, fmap, R, t)
    torch.testing.assert_close(_velocity_from_map(model, torch.flip(fmap, dims=(3,)), FLIP @ R @ FLIP, t), S * v,
                               atol=1e-8, rtol=1e-6)


def test_flip_breaks_without_symmetrisation_but_roll_holds():
    model = _flow(flip_symmetrize=False)
    fmap, R, t = _inputs(model, 2)
    v = _velocity_from_map(model, fmap, R, t)
    torch.testing.assert_close(_velocity_from_map(model, torch.rot90(fmap, 1, dims=(2, 3)), R_Z_MINUS_90 @ R, t), v,
                               atol=1e-8, rtol=1e-6)
    assert (_velocity_from_map(model, torch.flip(fmap, dims=(3,)), FLIP @ R @ FLIP, t) - S * v).abs().max() > 1e-3


def test_rotorcorr_flow_trains_and_samples():
    model = _flow(torch.float32).train()
    with torch.no_grad():
        model.vector_field.out.weight.zero_()
        model.vector_field.out.bias.zero_()
    img = torch.randn(2, 3, 224, 224)
    loss = model.compute_loss(img, rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    assert model.vector_field.out.weight.grad.abs().max() > 1e-6       # the zero read-out still learns
    assert model.vector_field.p.grad is None or torch.isfinite(model.vector_field.p.grad).all()
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)


def test_rotorcorr_needs_sphere_tokens():
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    with pytest.raises(ValueError):
        CliffordFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="rotorcorr", cond_tokens="pooled", condition_head="none")


def test_small_size_matches_the_other_heads():
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50",
                         conv_adapter=False, vector_field="rotorcorr", cond_tokens="sphere", condition_head="none",
                         rotorcorr=dict(c_s=32, templates=48, hidden=103))
    n = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("adapter.backbone"))
    assert abs(n - 248_723) <= 0.01 * 248_723, n    # DiT-small; SO(3)-Fourier small is 246,499
