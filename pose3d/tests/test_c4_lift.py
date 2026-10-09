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


def test_turn_matrix_is_r_z_minus_90():
    # The equivariance tests pass for any G with G^4 = I (e.g. G^T), so pin the sign separately:
    # a counter-clockwise turn on screen (x right, y down) is R_z(-90 deg) in the camera frame,
    # measured on the warp pipeline (probe_roll: sign -1 gives 8.05 deg at 5 deg, sign +1 13.2).
    from pose3d.models.c4_canon import G_TURN

    a = -math.pi / 2
    r_z = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(G_TURN, r_z, atol=1e-7, rtol=0)
    torch.testing.assert_close(G_TURN.double(), R_Z_MINUS_90)
    # a feature right of the centre (column offset +1) moves to the top (row offset -1) under rot90
    m = torch.zeros(1, 1, 3, 3)
    m[0, 0, 1, 2] = 1.0
    assert rot90(m, 1)[0, 0, 0, 1] == 1.0
    assert torch.allclose(G_TURN @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))


def test_eval_rotations_with_coupled_noise_gives_the_same_error_image_by_image(tmp_path, monkeypatch):
    # The real evaluation path on a c4lift + frame checkpoint: with the noise coupled (r0 -> g r0)
    # the error of every test image is the same at 0 and at 90/180/270 deg (Lean:
    # equivariant_error_invariant + euler_equivariant); 45 deg is not covered by the guarantee.
    import json
    import sys

    import numpy as np

    import pose3d.eval_rotations as ev

    model = _flow(dtype=torch.float32)
    ckpt = tmp_path / "c4.pth"
    torch.save({"model": model.state_dict(),
                "config": {"model": {"name": "clifford_flow", "encoder": "resnet50", "pretrained_backbone": False,
                                     "flow_hidden_dim": [8], "n_cond_mv": 6, "conv_adapter": False,
                                     "vector_field": "gatr", "gatr_blocks": 2, "gatr_mv_channels": 4,
                                     "gatr_s_channels": 8, "gatr_heads": 2, "cond_tokens": "c4lift",
                                     "so2_channels": 8, "pose_tokens": "frame"}}}, ckpt)

    class Fake(torch.utils.data.Dataset):
        def __init__(self, *a, **k):
            g = torch.Generator().manual_seed(3)
            self.img = torch.rand(4, 3, 224, 224, generator=g)
            self.rot = torch.linalg.qr(torch.randn(4, 3, 3, generator=g))[0]
            self.rot = self.rot * torch.linalg.det(self.rot).sign().view(-1, 1, 1)

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return dict(img=self.img[i], rot=self.rot[i], cls=torch.tensor([i % 2]))

    monkeypatch.setattr(ev, "Pascal3D", Fake)
    monkeypatch.setattr(ev, "get_available_device", lambda: torch.device("cpu"))
    out = tmp_path / "rot.json"
    monkeypatch.setattr(sys, "argv", ["x", "--checkpoint", str(ckpt), "--path_to_datasets", "unused",
                                      "--batch_size", "2", "--num_workers", "0", "--steps", "2",
                                      "--eval_samples", "3", "--angles", "0", "90", "180", "270", "45",
                                      "--couple_noise", "--output_json", str(out)])
    ev.main()
    per = np.load(str(out).replace(".json", "_per_sample.npz"))
    for deg in (90, 180, 270):
        np.testing.assert_allclose(per[f"err_{deg}"], per["err_0"], atol=2e-2)
    assert per["err_45"].shape == per["err_0"].shape   # runs; an untrained field barely sees the image
    assert set(json.load(open(out))["angles"]) == {"0", "90", "180", "270", "45"}


def test_roll_prior_resolves_the_four_way_turn_ambiguity():
    # An exactly C4-equivariant flow on an ambiguous crop puts its samples in the four turned modes;
    # the explicit upright prior (fitted on labels) makes the medoid pick the upright one.
    from pose3d.models.c4_canon import G_TURN
    from pose3d.models.clifford_flow import CliffordFlow, RollPrior

    torch.manual_seed(0)
    # labels: object z axis pointing image-up (-e2) with ~10 deg of roll jitter, random azimuth
    az = torch.rand(400) * 2 * math.pi
    roll = torch.randn(400) * math.radians(10)

    def rz(a):
        c, s = torch.cos(a), torch.sin(a)
        z, o = torch.zeros_like(a), torch.ones_like(a)
        return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)

    to_up = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # object z -> camera -e2
    labels = rz(roll) @ to_up @ rz(az)
    prior = RollPrior.fit(labels)
    assert torch.allclose(prior.axis.abs(), torch.tensor([0.0, 0.0, 1.0]))
    assert abs(prior.mu + math.pi / 2) < 0.05 and prior.kappa > 10

    flow = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], n_cond_mv=4, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False)
    from pose3d.geometry.rotor import matrix_to_rotor
    upright = labels[:1]
    modes = torch.cat([torch.linalg.matrix_power(G_TURN, k) @ upright for k in (1, 1, 1, 2, 2, 3, 0)])
    rotors = matrix_to_rotor(modes).unsqueeze(0)                      # (1, 7, 4): the upright mode is rarest
    plain = flow._medoid(rotors)
    weighted = flow._medoid(rotors, prior.weights(modes).unsqueeze(0))
    assert not torch.allclose(plain.abs(), rotors[0, -1].abs(), atol=1e-4)
    assert torch.allclose(weighted.abs(), rotors[0, -1].abs(), atol=1e-4)


def test_backbone_init_from_a_pooled_rotor_teacher(tmp_path):
    # Variant A starts from the 8.95 deg teacher's fine-tuned ResNet only: its heads were trained on
    # inputs of another meaning (pooled tokens, a rotor token), so they start fresh with a zero
    # read-out; the backbone and the fresh heads get separate learning rates, decided by name.
    from pose3d.config import Config
    from pose3d.models.clifford_flow import CliffordFlow
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    from pose3d.train import build_scheduler, init_from_checkpoint, param_groups

    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    gatr = dict(num_blocks=1, mv_channels=4, s_channels=8, num_heads=1)
    common = dict(hidden_dim=[8], n_cond_mv=4, pretrained_backbone=False, encoder_type="resnet50",
                  conv_adapter=False, vector_field="gatr", gatr=gatr)
    torch.manual_seed(0)
    teacher = CliffordFlow(CliffordAlgebra((1, 1, 1)), **common)
    with torch.no_grad():
        for p in teacher.vector_field.out.parameters():
            p.normal_()                                    # a trained teacher has a non-zero read-out
    path = tmp_path / "teacher.pth"
    torch.save({"model": teacher.state_dict()}, path)
    student = CliffordFlow(CliffordAlgebra((1, 1, 1)), cond_tokens="c4lift", so2_channels=8,
                           pose_tokens="frame", **common)
    cfg = Config()
    cfg.device = torch.device("cpu")
    cfg.flow.init_from, cfg.flow.init_mode, cfg.flow.backbone_lr_mult = str(path), "backbone", 0.35
    cfg.train.n_epochs, cfg.train.warmup_epochs = 10, 1
    init_from_checkpoint(student, cfg)
    teacher_sd, student_sd = teacher.state_dict(), student.state_dict()
    own_bb = {k: v for k, v in student_sd.items() if k.startswith("adapter.backbone.")}
    assert own_bb and all(torch.equal(v, teacher_sd[k]) for k, v in own_bb.items())
    assert all(p.abs().max() == 0 for p in student.vector_field.out.parameters())
    shared_heads = [k for k in student_sd if k.startswith("condition_head.") and k in teacher_sd
                    and student_sd[k].shape == teacher_sd[k].shape]
    assert shared_heads and not all(torch.equal(student_sd[k], teacher_sd[k]) for k in shared_heads)

    groups = param_groups(student, cfg)
    assert len(groups) == 2 and groups[0]["lr"] == pytest.approx(cfg.effective_lr * 0.35)
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in student.parameters())
    opt = torch.optim.AdamW(groups, lr=cfg.effective_lr)
    sched = build_scheduler(opt, cfg)
    peaks = [g["lr"] / 0.1 for g in opt.param_groups]       # warm-up starts at 10% of each peak
    for _ in range(cfg.train.n_epochs):
        opt.step()
        sched.step()
    for g, peak in zip(opt.param_groups, peaks):             # each group decays to 5% of its own peak
        assert g["lr"] == pytest.approx(0.05 * peak)

    bad = tmp_path / "bad.pth"
    torch.save({"model": {k: v for k, v in teacher_sd.items() if not k.endswith("conv1.weight")}}, bad)
    cfg.flow.init_from = str(bad)
    with pytest.raises(ValueError):
        init_from_checkpoint(student, cfg)
