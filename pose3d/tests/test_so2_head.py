"""cond_tokens='so2' + pose_tokens='frame': the condition tokens and the whole velocity field respect
in-plane rotations of the image (exactly, for 90 degree turns of the backbone map)."""

import math

import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.geometry.flow import rotor_multiply
from pose3d.geometry.rotor import matrix_to_rotor, random_rotor
from pose3d.models.so2_head import SO2ConditionHead

# torch.rot90(x, 1, dims=(2, 3)) turns the image counter-clockwise on screen; the warp pipeline
# maps the pose label R to R_z(-90 deg) R for that turn (x right, y down, z along the optical axis).
R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _rot90(fmap):
    return torch.rot90(fmap, 1, dims=(2, 3))


def test_head_tokens_rotate_with_the_map():
    torch.manual_seed(0)
    head = SO2ConditionHead(c_in=32, n_out=9, channels=16).double()
    with torch.no_grad():
        head.radial.normal_()
    fmap = torch.randn(3, 32, 7, 7, dtype=torch.float64)
    mv, mv_rot = head(fmap), head(_rot90(fmap))
    r2 = R_Z_MINUS_90[:2, :2]
    torch.testing.assert_close(mv_rot[..., [0, 3, 4, 7]], mv[..., [0, 3, 4, 7]])
    torch.testing.assert_close(mv_rot[..., [1, 2]], mv[..., [1, 2]] @ r2.T)
    torch.testing.assert_close(mv_rot[..., [5, 6]], mv[..., [5, 6]] @ r2.T)
    assert mv[:, :-1, [1, 2]].abs().max() > 1e-3            # the vector parts are not trivially 0
    axis = torch.zeros(8, dtype=torch.float64)
    axis[3] = 1.0
    torch.testing.assert_close(mv[:, -1], axis.expand(3, 8))


def test_up_token_is_constant_and_image_tokens_still_rotate():
    torch.manual_seed(0)
    head = SO2ConditionHead(c_in=32, n_out=9, channels=16, up_token=True).double()
    with torch.no_grad():
        head.radial.normal_()
    fmap = torch.randn(3, 32, 7, 7, dtype=torch.float64)
    mv, mv_rot = head(fmap), head(_rot90(fmap))
    assert mv.shape == (3, 9, 8)
    r2 = R_Z_MINUS_90[:2, :2]
    img, img_rot = mv[:, :-2], mv_rot[:, :-2]
    torch.testing.assert_close(img_rot[..., [1, 2]], img[..., [1, 2]] @ r2.T)
    torch.testing.assert_close(img_rot[..., [0, 3, 4, 7]], img[..., [0, 3, 4, 7]])
    axis, up = torch.zeros(8, dtype=torch.float64), torch.zeros(8, dtype=torch.float64)
    axis[3], up[2] = 1.0, -1.0
    torch.testing.assert_close(mv[:, -2], axis.expand(3, 8))
    torch.testing.assert_close(mv[:, -1], up.expand(3, 8))          # does not turn with the image
    torch.testing.assert_close(mv_rot[:, -1], up.expand(3, 8))


def test_head_rejects_a_wrong_map_size():
    with pytest.raises(ValueError):
        SO2ConditionHead(c_in=8, n_out=4, channels=4)(torch.randn(1, 8, 5, 5))


def _flow(pose_tokens, dtype=torch.float64):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    # A fresh algebra per model: .to(dtype) converts its tables in place.
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], n_cond_mv=6, pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="so2", so2_channels=8,
                         pose_tokens=pose_tokens,
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).to(dtype)
    with torch.no_grad():  # undo the zero init of the read-out and the flat radial profiles
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
        model.adapter.conv_adapter.radial.normal_()
    return model.eval()


def _velocity_from_map(model, fmap, rotor, t):
    cond = model.condition_head(model.adapter.conv_adapter(fmap))
    return model.velocity(rotor, t, cond)


def test_frame_velocity_is_invariant_to_rotating_image_and_pose_together():
    # Turning the image turns the true pose: R -> G R. The body-frame velocity the flow integrates
    # (r <- r exp(dt v)) must then be unchanged.
    model = _flow("frame")
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    rotor = random_rotor(2).double()
    g = matrix_to_rotor(R_Z_MINUS_90).expand(2, 4)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, rotor, t)
    v_rot = _velocity_from_map(model, _rot90(fmap), rotor_multiply(g, rotor, model.algebra), t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_rot, v, atol=1e-8, rtol=1e-6)


def test_rotor_token_breaks_that_invariance():
    # Negative control: GATr conjugates a rotor token, so the same check fails with pose_tokens=rotor.
    model = _flow("rotor")
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    rotor = random_rotor(2).double()
    g = matrix_to_rotor(R_Z_MINUS_90).expand(2, 4)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, rotor, t)
    v_rot = _velocity_from_map(model, _rot90(fmap), rotor_multiply(g, rotor, model.algebra), t)
    assert (v_rot - v).abs().max() > 1e-4


def test_so2_frame_flow_trains_and_samples():
    model = _flow("frame", dtype=torch.float32).train()
    img = torch.randn(2, 3, 224, 224)
    rot = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    rot = rot * torch.linalg.det(rot).sign().view(-1, 1, 1)
    loss = model.compute_loss(img, rot)
    loss.backward()
    head_grads = [p.grad for p in model.adapter.conv_adapter.parameters() if p.grad is not None]
    assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
    assert math.isfinite(loss.item())
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)
