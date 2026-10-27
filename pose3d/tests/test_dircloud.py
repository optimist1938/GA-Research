"""cond_tokens='dircloud' + pose_tokens='frame_pin' (P1): the direction cloud and the whole velocity field
are exactly equivariant to 90 degree turns and to left-right flips of the backbone map."""

import math

import pytest
import torch

from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix
from pose3d.models.dircloud_head import DirectionCloudHead

# torch.rot90(x, 1, dims=(2, 3)) turns the image counter-clockwise on screen: label R -> R_z(-90 deg) R
# (see test_so2_head.py). A left-right flip of the map is F = diag(-1, 1, 1): label R -> F R F.
R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
FLIP = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))


def _rot90(fmap):
    return torch.rot90(fmap, 1, dims=(2, 3))


def _flip(fmap):
    return torch.flip(fmap, dims=(3,))


def _cell_perm(transform, grid=7):
    """Index of each original cell in the transformed map."""
    idx = torch.arange(grid * grid, dtype=torch.float64).view(1, 1, grid, grid)
    moved = transform(idx).flatten().long()          # moved[k] = original cell now at position k
    perm = torch.empty_like(moved)
    perm[moved] = torch.arange(grid * grid)
    return perm                                      # perm[j] = new position of original cell j


@pytest.mark.parametrize("transform,G", [(_rot90, R_Z_MINUS_90), (_flip, FLIP)])
def test_cloud_moves_with_the_map(transform, G):
    torch.manual_seed(0)
    head = DirectionCloudHead(c_in=32, scalars=16).double()
    fmap = torch.randn(2, 32, 7, 7, dtype=torch.float64)
    out, out_t = head(fmap), head(transform(fmap))
    perm = _cell_perm(transform)
    cells, cells_t = out[:, :-1], out_t[:, :-1][:, perm]
    torch.testing.assert_close(cells_t[..., 8:], cells[..., 8:])                 # scalars ride along
    torch.testing.assert_close(cells_t[..., 1:4], cells[..., 1:4] @ G.T)         # directions turn
    torch.testing.assert_close(out_t[:, -1], out[:, -1])                         # e3 axis token
    torch.testing.assert_close(head.direction.norm(dim=-1), torch.ones(49, dtype=head.direction.dtype))


def _flow(dtype=torch.float64, up_token=False, final_norm=False):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="dircloud", pose_tokens="frame_pin",
                         condition_head="none", dircloud_scalars=8, dircloud_up_token=up_token,
                         gatr_final_norm=final_norm,
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).to(dtype)
    with torch.no_grad():   # undo the zero-initialised read-out
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
    return model.eval()


def _velocity_from_map(model, fmap, rotor, t):
    return model.velocity(rotor, t, model.condition_head(model.adapter.conv_adapter(fmap)))


def test_velocity_invariant_to_turning_image_and_pose_together():
    model = _flow()
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    v_rot = _velocity_from_map(model, _rot90(fmap), matrix_to_rotor(R_Z_MINUS_90 @ R), t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_rot, v, atol=1e-8, rtol=1e-6)


def test_velocity_flips_with_a_mirrored_image():
    # Flipped image, pose F R F: the body velocity must be F v F, i.e. (e12, e13, e23) -> (-, -, +).
    model = _flow()
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.2, 0.9], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    v_flip = _velocity_from_map(model, _flip(fmap), matrix_to_rotor(FLIP @ R @ FLIP), t)
    sign = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_flip, sign * v, atol=1e-8, rtol=1e-6)


def test_initial_field_gets_a_gradient_through_the_bivector_read_out():
    # The frame-token runs stalled when the read-out grade had no signal at the input; here the
    # axial pose channel is a bivector, so the zero-initialised read-out must still get a gradient.
    model = _flow(torch.float32)
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.zero_()
    model.train()
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    g = model.vector_field.out.weight.grad if hasattr(model.vector_field.out, "weight") else \
        next(p.grad for p in model.vector_field.out.parameters())
    assert g is not None and g.abs().max() > 1e-6


def test_dircloud_flow_trains_and_samples():
    model = _flow(torch.float32).train()
    assert model.n_cond_mv == 50
    img = torch.randn(2, 3, 224, 224)
    rot = rotor_to_matrix(random_rotor(2), model.algebra)
    loss = model.compute_loss(img, rot)
    loss.backward()
    assert math.isfinite(loss.item())
    assert all(torch.isfinite(p.grad).all() for p in model.adapter.conv_adapter.parameters() if p.grad is not None)
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)


def test_dircloud_needs_its_pose_tokens():
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    with pytest.raises(ValueError):
        CliffordFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="gatr", cond_tokens="dircloud",
                     pose_tokens="frame_ch", condition_head="none")


def _p1_flow(zero_readout):
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    return CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                        vector_field="gatr", cond_tokens="dircloud", pose_tokens="frame_pin",
                        condition_head="none", dircloud_scalars=8, gatr_zero_readout=zero_readout,
                        gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2))


def test_zero_readout_leaves_no_gradient_upstream():
    """The SYMSOL stall: with the zeroed read-out nothing upstream of it learns, only its few bivector weights,
    which a uniform pose prior averages to zero. GATr's own read-out init lets every layer learn from step 0."""
    for zero, upstream_learns in ((True, False), (False, True)):
        model = _p1_flow(zero).train()
        model.compute_loss(torch.rand(2, 3, 224, 224), rotor_to_matrix(random_rotor(2), model.algebra)).backward()
        blocks = [p.grad for n, p in model.vector_field.named_parameters() if "linear_out" not in n]
        moved = any(g is not None and g.abs().max() > 0 for g in blocks)
        assert moved == upstream_learns


def test_up_token_keeps_the_flip_and_breaks_the_turn():
    """--dircloud_up_token: the constant -e2 token is unchanged by the left-right flip, so the flow still flips
    with a mirrored image exactly; a 90 deg turn now changes the answer (the head may read the channels)."""
    model = _flow(up_token=True)
    assert model.n_cond_mv == 51
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.2, 0.9], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    v_flip = _velocity_from_map(model, _flip(fmap), matrix_to_rotor(FLIP @ R @ FLIP), t)
    torch.testing.assert_close(v_flip, torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64) * v,
                               atol=1e-8, rtol=1e-6)
    v_rot = _velocity_from_map(model, _rot90(fmap), matrix_to_rotor(R_Z_MINUS_90 @ R), t)
    # Small at random init (little reaches the pose token yet), but far above float64 rounding (~1e-13 when exact).
    assert (v_rot - v).abs().max() > 1e-9


def test_up_token_layout():
    from pose3d.models.dircloud_head import DirectionCloudHead

    head = DirectionCloudHead(c_in=16, scalars=4, up_token=True)
    out = head(torch.randn(1, 16, 7, 7))
    assert out.shape == (1, 51, 8 + 4 + 2)
    assert out[0, 49, 3] == 1 and out[0, 49, 12] == 1 and out[0, 49, 13] == 0     # axis e3 + its flag
    assert out[0, 50, 2] == -1 and out[0, 50, 13] == 1 and out[0, 50, 12] == 0    # up -e2 + its flag


def test_final_norm_keeps_both_symmetries_exact():
    model = _flow(final_norm=True)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity_from_map(model, _rot90(fmap), matrix_to_rotor(R_Z_MINUS_90 @ R), t), v,
                               atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(_velocity_from_map(model, _flip(fmap), matrix_to_rotor(FLIP @ R @ FLIP), t),
                               torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64) * v, atol=1e-8, rtol=1e-6)


def test_final_norm_read_out_ignores_the_hidden_scale():
    """The route P1 used to switch shapes off (shrinking the hidden state) no longer changes the output."""
    model = _flow(final_norm=True)
    read_out = model.vector_field.net.linear_out
    mv = torch.randn(3, 5, 4, 16, dtype=torch.float64)
    s = torch.randn(3, 5, 8, dtype=torch.float64)
    a_mv, a_s = read_out(mv, scalars=s)
    b_mv, b_s = read_out(20.0 * mv, scalars=20.0 * s)
    torch.testing.assert_close(b_mv, a_mv, atol=1e-3, rtol=1e-3)
