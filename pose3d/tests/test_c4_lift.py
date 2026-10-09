"""cond_tokens='c4lift': a C4-lifted backbone + harmonic head makes the whole velocity field respect
90-degree turns of the *image* (not just of the backbone map, as the SO(2) head does)."""

import math

import pytest
import torch
import torch.nn as nn
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.geometry.flow import rotor_multiply
from pose3d.geometry.rotor import matrix_to_rotor, random_rotor
from pose3d.models.c4_lift import C4HarmonicHead, c4_lift, rot90
from pose3d.models.so2_head import SO2ConditionHead

# torch.rot90(x, 1, dims=(2, 3)) turns the image counter-clockwise on screen; the warp pipeline
# maps the pose label R to R_z(-90 deg) R for that turn (x right, y down, z along the optical axis).
R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _toy_backbone(c_out=16):
    # A plain (non-equivariant) CNN: 28x28 image -> 7x7 map.
    torch.manual_seed(1)
    return nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(8, c_out, 3, stride=2, padding=1), nn.ReLU()).double()


def _assert_tokens_turned(mv, mv_rot):
    r2 = R_Z_MINUS_90[:2, :2]
    torch.testing.assert_close(mv_rot[..., [0, 3, 4, 7]], mv[..., [0, 3, 4, 7]])
    torch.testing.assert_close(mv_rot[..., [1, 2]], mv[..., [1, 2]] @ r2.T)
    torch.testing.assert_close(mv_rot[..., [5, 6]], mv[..., [5, 6]] @ r2.T)


def _head(c_in=16, n_out=9):
    torch.manual_seed(0)
    head = C4HarmonicHead(c_in=c_in, n_out=n_out, channels=6).double()
    with torch.no_grad():
        head.radial.normal_()
    return head


def test_lift_is_the_regular_representation():
    bb = _toy_backbone()
    x = torch.randn(2, 3, 28, 28, dtype=torch.float64)
    f, f_rot = c4_lift(bb, x), c4_lift(bb, rot90(x, 1))
    for k in range(4):   # F_k(turned x) = turn(F_{k-1}(x))
        torch.testing.assert_close(f_rot[:, k], rot90(f[:, (k - 1) % 4], 1))


def test_tokens_turn_with_the_image():
    bb, head = _toy_backbone(), _head()
    x = torch.randn(3, 3, 28, 28, dtype=torch.float64)
    mv, mv_rot = head(c4_lift(bb, x)), head(c4_lift(bb, rot90(x, 1)))
    _assert_tokens_turned(mv, mv_rot)
    assert mv[:, :-1, [1, 2]].abs().max() > 1e-3            # vector parts are not trivially 0
    assert mv[:, :-1, [5, 6]].abs().max() > 1e-3
    for k in (2, 3):                                        # the other turns follow by composition
        mv_k = head(c4_lift(bb, rot90(x, k)))
        g = torch.linalg.matrix_power(R_Z_MINUS_90[:2, :2], k)
        torch.testing.assert_close(mv_k[..., [1, 2]], mv[..., [1, 2]] @ g.T)
    axis = torch.zeros(8, dtype=torch.float64)
    axis[3] = 1.0
    torch.testing.assert_close(mv[:, -1], axis.expand(3, 8))


def test_each_harmonic_alone_is_equivariant():
    # Zeroing all but one family of weights isolates c0 / c1 / c2: each must be equivariant alone,
    # which checks the sign conventions of every term, not just their sum.
    bb = _toy_backbone()
    x = torch.randn(2, 3, 28, 28, dtype=torch.float64)
    families = {"c0": ["from_c0"], "c1": ["c1_re", "c1_im"], "c2": ["c2_re", "c2_im"]}
    for keep in families.values():
        head = _head()
        with torch.no_grad():
            for name in ("from_c0", "c1_re", "c1_im", "c2_re", "c2_im"):
                if name not in keep:
                    for p in getattr(head, name).parameters():
                        p.zero_()
        mv, mv_rot = head(c4_lift(bb, x)), head(c4_lift(bb, rot90(x, 1)))
        assert mv[:, :-1].abs().max() > 1e-3
        _assert_tokens_turned(mv, mv_rot)


def test_so2_head_on_a_plain_backbone_does_not_turn_with_the_image():
    # Negative control: the SO(2) head is equivariant to turning its *map*, but a plain CNN's map
    # does not turn with the image, so the tokens of a turned image are not the turned tokens.
    bb = _toy_backbone()
    torch.manual_seed(0)
    head = SO2ConditionHead(c_in=16, n_out=9, channels=6).double()
    with torch.no_grad():
        head.radial.normal_()
    x = torch.randn(3, 3, 28, 28, dtype=torch.float64)
    mv, mv_rot = head(bb(x)), head(bb(rot90(x, 1)))
    r2 = R_Z_MINUS_90[:2, :2]
    assert (mv_rot[..., [1, 2]] - mv[..., [1, 2]] @ r2.T).abs().max() > 1e-3


def _flow(dtype=torch.float64):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], n_cond_mv=6, pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="c4lift", so2_channels=8,
                         pose_tokens="frame",
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).to(dtype)
    with torch.no_grad():  # undo the zero init of the read-out and the flat radial profiles
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
        model.adapter.conv_adapter.radial.normal_()
    return model.eval()


def test_velocity_is_invariant_to_turning_image_and_pose_together():
    # End to end, from pixels: turning the image turns the true pose, R -> G R, and the body-frame
    # velocity the flow integrates (r <- r exp(dt v)) is unchanged (Lean: euler_equivariant).
    model = _flow()
    img = torch.randn(2, 3, 224, 224, dtype=torch.float64)
    rotor = random_rotor(2).double()
    g = matrix_to_rotor(R_Z_MINUS_90).expand(2, 4)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    with torch.no_grad():
        v = model.velocity(rotor, t, model.condition(img))
        v_rot = model.velocity(rotor_multiply(g, rotor, model.algebra), t, model.condition(rot90(img, 1)))
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_rot, v, atol=1e-8, rtol=1e-6)


def test_c4lift_flow_trains_and_samples():
    model = _flow(dtype=torch.float32).train()
    img = torch.randn(2, 3, 224, 224)
    rot = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    rot = rot * torch.linalg.det(rot).sign().view(-1, 1, 1)
    loss = model.compute_loss(img, rot)
    loss.backward()
    head_grads = [p.grad for p in model.adapter.conv_adapter.parameters() if p.grad is not None]
    assert head_grads and all(torch.isfinite(g).all() for g in head_grads)
    assert math.isfinite(loss.item())
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)


def test_c4_canonicalized_model_is_exactly_equivariant_and_keeps_upright_predictions():
    # Variant B: argmax-over-the-orbit canonicaliser around an arbitrary (here: untrained, pooled,
    # rotor-token) flow. Same sampling noise on both calls, so the check is exact, sample by sample.
    from pose3d.models.c4_canon import G_TURN, C4Canonicalized
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    flow = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], n_cond_mv=4, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False).eval()
    model = C4Canonicalized(flow, scorer_weight=torch.randn(2048)).eval()
    img = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        c = model.canonical_turn(img)
        torch.manual_seed(5)
        rot = model.predict(img, n_samples=2, steps=2)
        torch.manual_seed(5)
        rot_turned = model.predict(rot90(img, 1), n_samples=2, steps=2)
        assert torch.equal(model.canonical_turn(rot90(img, 1)), (c + 1) % 4)
        torch.testing.assert_close(rot_turned, G_TURN @ rot, atol=1e-5, rtol=1e-5)
        # where the scorer picks the upright view, the wrapper is the model itself
        upright = img[c == 0]
        if len(upright):
            torch.manual_seed(7)
            a = model.predict(upright, n_samples=2, steps=2)
            torch.manual_seed(7)
            b = flow.predict(upright, n_samples=2, steps=2)
            torch.testing.assert_close(a, b)


def test_partial_reflow_init_from_a_pooled_rotor_teacher(tmp_path):
    # Variant A is trained by reflow from the 8.95 deg teacher, whose architecture differs (pooled
    # tokens, rotor token): matching tensors are copied, the new head / first GATr layer start fresh
    # and get their own learning rate.
    from pose3d.config import Config
    from pose3d.models.clifford_flow import CliffordFlow
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    from pose3d.train import attach_reflow_teacher, param_groups

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    gatr = dict(num_blocks=1, mv_channels=4, s_channels=8, num_heads=1)
    common = dict(hidden_dim=[8], n_cond_mv=4, pretrained_backbone=False, encoder_type="resnet50",
                  conv_adapter=False, vector_field="gatr", gatr=gatr)
    torch.manual_seed(0)
    teacher = CliffordFlow(CliffordAlgebra((1, 1, 1)), **common)
    path = tmp_path / "teacher.pth"
    torch.save({"model": teacher.state_dict(),
                "config": {"model": {"name": "clifford_flow", "encoder": "resnet50", "pretrained_backbone": False,
                                     "flow_hidden_dim": [8], "n_cond_mv": 4, "conv_adapter": False,
                                     "vector_field": "gatr", "gatr_blocks": 1, "gatr_mv_channels": 4,
                                     "gatr_s_channels": 8, "gatr_heads": 1}}}, path)
    student = CliffordFlow(CliffordAlgebra((1, 1, 1)), cond_tokens="c4lift", so2_channels=8,
                           pose_tokens="frame", **common)
    cfg = Config()
    cfg.device = torch.device("cpu")
    cfg.flow.reflow_teacher, cfg.flow.reflow_init, cfg.flow.fresh_lr_mult = str(path), "partial", 10.0
    attach_reflow_teacher(student, cfg)
    teacher_sd = teacher.state_dict()
    own_bb = {k: v for k, v in student.state_dict().items() if k.startswith("adapter.backbone")}
    assert own_bb and all(torch.equal(v, teacher_sd[k]) for k, v in own_bb.items())
    assert any(n.startswith("adapter.conv_adapter") for n in student._fresh_params)
    groups = param_groups(student, cfg)
    assert len(groups) == 2 and groups[1]["lr"] == cfg.effective_lr * 10.0
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in student.parameters())
