"""SYMSOL: the loader (fresh equivalents in training, all of them at test), the nearest-equivalent error, and
the evaluation + log-likelihood path of the two flows compared on it (DiTFlow, P1's GATr CliffordFlow)."""

import math
import os

import numpy as np
import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from PIL import Image
from torch.utils.data import DataLoader

import pose3d.geometry.matrix as matrix_flow
from pose3d.datasets.symsol import MAX_EQUIVALENTS, Symsol, to_camera_frame
from pose3d.engine.likelihood import LOG_UNIFORM, flow_log_likelihood, symsol_log_likelihood
from pose3d.engine.metrics import nearest_equivalent_error

SHAPES = {"tet": 12, "cube": 24, "icosa": 60, "cone": 360, "cyl": 720}


def _fake_symsol(root, n_views=3, size=224):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        path = os.path.join(root, "symsol", split)
        os.makedirs(os.path.join(path, "images"))
        rots = {}
        for name, k in SHAPES.items():
            rots[name] = matrix_flow.exp_map(torch.tensor(rng.normal(size=(n_views, k, 3)))).numpy()
            for i in range(n_views):
                img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
                Image.fromarray(img).save(os.path.join(path, "images", f"{name}_{i:05d}.png"))
        np.savez(os.path.join(path, "rotations.npz"), **rots)
    return root


def test_loader_draws_equivalents_and_pads_the_test_set(tmp_path):
    root = _fake_symsol(str(tmp_path))
    train = Symsol(root, train=True, num_views=2)
    test = Symsol(root, train=False)
    assert len(train) == 10 and len(test) == 15
    raw = np.load(os.path.join(root, "symsol", "train", "rotations.npz"))
    cone = to_camera_frame(raw["cone"][1])                                  # train index 7 = cone, view 1
    draws = torch.stack([train[7]["rot"] for _ in range(20)])
    assert len({tuple(d.flatten().tolist()) for d in draws}) > 1               # a fresh equivalent per read
    assert all(np.abs(cone - d.numpy()).sum((1, 2)).min() < 1e-6 for d in draws)
    item = test[3]                                                            # cube, view 0
    assert item["rots_eq"].shape == (MAX_EQUIVALENTS, 3, 3) and int(item["n_eq"]) == 24
    assert item["cls_eval"].tolist() == [1] and item["img"].dtype == torch.float32
    torch.testing.assert_close(item["rot"], item["rots_eq"][0])


def test_nearest_equivalent_error_ignores_padding():
    R = matrix_flow.exp_map(torch.tensor([[0.0, 0.0, 0.3]]))
    rots = torch.zeros(1, 4, 3, 3)
    rots[0, 0] = matrix_flow.exp_map(torch.tensor([0.0, 0.0, 1.3]))
    rots[0, 1] = matrix_flow.exp_map(torch.tensor([0.0, 0.0, 0.5]))
    rots[0, 2] = R[0]                                                          # padding: must not count
    err = nearest_equivalent_error(R, rots, torch.tensor([2]))
    assert abs(err[0] - math.degrees(0.2)) < 1e-3


def _dit():
    from pose3d.models.dit_flow import DiTFlow
    torch.manual_seed(0)
    return DiTFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50",
                   n_time_samples=2, dim=32, blocks=2, heads=2).eval()


def _p1(scale=0.5):
    from pose3d.models.clifford_flow import CliffordFlow
    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="dircloud", pose_tokens="frame_pin",
                         condition_head="none", dircloud_scalars=8,
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2))
    with torch.no_grad():   # undo the zero-initialised read-out
        for p in model.vector_field.out.parameters():
            p.normal_(0, scale)
    return model.eval()


def _mlp():
    from pose3d.models.mlp_flow import MLPFlow
    torch.manual_seed(0)
    return MLPFlow(CliffordAlgebra((1, 1, 1)), cond_hidden=16, cond_features=8, field_hidden=16,
                   encoder_type="resnet50", conv_adapter=False).eval()


@pytest.mark.parametrize("make", [_dit, _p1, _mlp])
def test_symsol_evaluation_end_to_end(tmp_path, make):
    from pose3d.engine.metrics import calculate_evaluation_metrics

    loader = DataLoader(Symsol(_fake_symsol(str(tmp_path), n_views=1), train=False), batch_size=3)
    model = make()

    class Cfg:
        device = torch.device("cpu")
        rank = 0
        features = type("F", (), {"fixed_val_noise": True})()
        run = type("R", (), {"platform": "kaggle"})()

    err, cls = calculate_evaluation_metrics(model, loader, Cfg(), n_samples=2, return_classes=True)
    assert err.shape == (5,) and np.isfinite(err).all() and (err <= 180).all()
    assert cls.tolist() == [0, 1, 2, 3, 4]
    ll, ll_cls = symsol_log_likelihood(model, loader, "cpu", n_equivalents=2, steps=2)
    assert ll.shape == (5,) and np.isfinite(ll).all() and ll_cls.tolist() == [0, 1, 2, 3, 4]
    if make in (_dit, _mlp):   # zero-initialised output: the untrained flow is the identity, so the uniform density
        np.testing.assert_allclose(ll, LOG_UNIFORM, atol=1e-6)


def _dit_perturbed(scale=0.1):
    model = _dit()
    with torch.no_grad():   # leave the identity flow
        for p in model.vector_field.parameters():
            p.add_(scale * torch.randn_like(p))
    return model


@pytest.mark.parametrize("make", [_dit_perturbed, lambda: _p1(5.0)])
def test_divergence_matches_finite_differences(make):
    """The autograd divergence through the first-order retraction, against central differences of exp."""
    from pose3d.engine.likelihood import _velocity_and_divergence, flow_states

    model = make()
    states = flow_states(model)
    torch.manual_seed(2)
    cond = model.condition(torch.rand(1, 3, 224, 224))
    cond = cond.expand(6, *cond.shape[1:])
    s = states.from_matrix(matrix_flow.random_rotation_matrices(6))
    t = torch.full((6,), 0.4)
    _, div = _velocity_and_divergence(model, states, s, t, cond)
    h, fd = 1e-3, torch.zeros(6)
    with torch.no_grad():
        for i in range(3):
            e = torch.zeros(6, 3)
            e[:, i] = h
            fd += (model.velocity(states.step(s, e), t, cond)[:, i]
                   - model.velocity(states.step(s, -e), t, cond)[:, i]) / (2 * h)
    torch.testing.assert_close(div, fd, atol=1e-3, rtol=2e-2)


def test_density_integrates_to_one():
    """E_uniform[p / p_uniform] = 1 for a non-trivial learned-looking field (perturbed DiT)."""
    model = _dit_perturbed()
    torch.manual_seed(1)
    cond = model.condition(torch.rand(1, 3, 224, 224))
    n = 4000
    ll = flow_log_likelihood(model, cond.expand(n, *cond.shape[1:]), matrix_flow.random_rotation_matrices(n),
                             steps=32)
    ratio = torch.exp(ll - LOG_UNIFORM)
    assert ratio.std() > 0.05                                    # a non-trivial density
    assert abs(ratio.mean().item() - 1) < 4 * ratio.std().item() / math.sqrt(n) + 0.02


R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _silhouette_direction(img):
    """Principal direction angle of the object's silhouette (x = columns right, y = rows down), mod 180 deg."""
    diff = (img - img[:, :1, :1]).abs().sum(0) > 0.12            # background = the corner colour
    ys, xs = torch.nonzero(diff, as_tuple=True)
    pts = torch.stack([xs, ys]).double()
    ev, evec = torch.linalg.eigh(torch.cov(pts))
    return torch.atan2(evec[1, 1], evec[0, 1])


def _axis_alignment(images, rots_eq, n_eq, frame=torch.eye(3)):
    """Weighted mean of cos 2(silhouette angle - projected cylinder axis angle): 1 when the axes agree."""
    num = den = 0.0
    for img, rots, n in zip(images, rots_eq, n_eq):
        R0 = frame @ rots[0]
        # The symmetry axis, in the camera frame: the rotation axis of R_90 R0^T (equivalents 0-359 turn the
        # cylinder about its axis in 1 degree steps, 360-719 also flip it end over end).
        Rk = frame @ rots[90]
        rel = Rk @ R0.T
        a = torch.stack([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        a = a / a.norm()
        w = 1 - a[2] ** 2
        num += w * torch.cos(2 * (_silhouette_direction(img) - torch.atan2(a[1], a[0])))
        den += w
    return float(num / den)


@pytest.mark.skipif(not os.environ.get("SYMSOL_ROOT"), reason="needs the SYMSOL data (SYMSOL_ROOT)")
def test_pose_frame_matches_the_images():
    """TO_CAMERA puts SYMSOL's poses in our camera frame (e1 columns, e2 rows down): the cylinder's projected
    symmetry axis lines up with its silhouette, also after a 90 degree turn of the image (R -> R_z(-90) R)."""
    test = Symsol(os.environ["SYMSOL_ROOT"], train=False, num_views=200)
    cyl = [test[i] for i in range(len(test)) if test.shapes[int(test.cls[i])] == "cyl"]
    images = [c["img"] for c in cyl]
    rots, n_eq = [c["rots_eq"] for c in cyl], [c["n_eq"] for c in cyl]
    straight = _axis_alignment(images, rots, n_eq)
    turned = _axis_alignment([torch.rot90(im, 1, dims=(1, 2)) for im in images], rots, n_eq, frame=R_Z_MINUS_90)
    y_flipped = _axis_alignment(images, rots, n_eq, frame=torch.diag(torch.tensor([1.0, -1.0, -1.0])))
    assert abs(straight) > 0.8 and abs(turned) > 0.8, (straight, turned)
    assert abs(y_flipped) < 0.5, y_flipped                       # the check can tell the frames apart


def _cube_group():
    """The 24 rotations of the cube (signed permutation matrices with det +1)."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1.0, -1.0], repeat=3):
            m = torch.zeros(3, 3)
            for r, (c, s) in enumerate(zip(perm, signs)):
                m[r, c] = s
            if torch.det(m) > 0:
                out.append(m)
    return torch.stack(out)


def test_nearest_equivalent_picks_the_closest_valid_pose():
    G = _cube_group()
    R1 = matrix_flow.random_rotation_matrices(1)[0]
    S = torch.zeros(1, 30, 3, 3)
    S[0, :24] = R1 @ G
    R0 = matrix_flow.random_rotation_matrices(1)
    near = matrix_flow.nearest_equivalent(R0, S, torch.tensor([24]))
    d = matrix_flow.geodesic_distance(R0.expand(24, 3, 3), S[0, :24])
    torch.testing.assert_close(matrix_flow.geodesic_distance(R0, near), d.min().view(1))
    S[0, 24] = R0[0]                                                  # padding equal to R0 must be ignored
    torch.testing.assert_close(matrix_flow.nearest_equivalent(R0, S, torch.tensor([24])), near)


def test_nearest_coupling_keeps_the_symmetric_distribution():
    """From a uniform R0 each equivalent is the nearest equally often, and the paths get much shorter."""
    G = _cube_group()
    n = 24000
    R0 = matrix_flow.random_rotation_matrices(n)
    S = G.expand(n, 24, 3, 3)
    near = matrix_flow.nearest_equivalent(R0, S, torch.full((n,), 24))
    k = (near.reshape(n, 1, 9) - G.reshape(1, 24, 9)).abs().sum(-1).argmin(1)
    freq = torch.bincount(k, minlength=24).float() / n
    assert (freq - 1 / 24).abs().max() < 0.012
    random_len = (matrix_flow.relative_log(R0, S[torch.arange(n), torch.randint(24, (n,))]) ** 2).sum(-1).mean()
    near_len = (matrix_flow.relative_log(R0, near) ** 2).sum(-1).mean()
    assert near_len < 0.2 * random_len


@pytest.mark.parametrize("make", [_dit, _p1])
def test_coupling_with_one_equivalent_is_the_plain_loss(make):
    model = make().train()
    img, R = torch.rand(2, 3, 224, 224), matrix_flow.random_rotation_matrices(2)
    rots = torch.zeros(2, 5, 3, 3)
    rots[:, 0] = R
    torch.manual_seed(3)
    plain = model.compute_loss(img, R)
    torch.manual_seed(3)
    coupled = model.compute_loss(img, R, rots_eq=rots, n_eq=torch.tensor([1, 1]))
    torch.testing.assert_close(coupled, plain)


def test_trainer_passes_equivalents_only_with_the_flag():
    from pose3d.engine.trainer import _compute_loss

    seen = {}

    class Model:
        def compute_loss(self, img, rot_gt, criterion=None, cls=None, rots_eq=None, n_eq=None):
            seen["rots_eq"] = rots_eq
            return torch.zeros(())

    data = dict(img=torch.zeros(1, 3, 4, 4), rot=torch.eye(3)[None], cls_eval=torch.zeros(1, 1),
                rots_eq=torch.eye(3).expand(1, 2, 3, 3), n_eq=torch.tensor([2]))
    for flag in (False, True):
        cfg = type("C", (), {"device": torch.device("cpu"),
                             "data": type("D", (), {"symsol_nearest_coupling": flag})()})()
        _compute_loss(Model(), data, None, cfg)
        assert (seen["rots_eq"] is not None) == flag


def test_image2sphere_recipe_optimizer_and_schedule():
    """--optimizer sgd --lr_schedule step: SGD with Nesterov momentum 0.9, lr x0.1 every 15 epochs (I2S / Howell)."""
    from pose3d.config import parse_args
    from pose3d.train import build_optimizer, build_scheduler

    cfg = parse_args(['--path_to_datasets', '/tmp', '--optimizer', 'sgd', '--lr', '1e-3', '--lr_scaling', 'none',
                      '--lr_schedule', 'step', '--warmup_epochs', '0', '--n_epochs', '40'])
    model = torch.nn.Linear(3, 3)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.SGD)
    g = opt.param_groups[0]
    assert g['lr'] == 1e-3 and g['momentum'] == 0.9 and g['nesterov'] and g['weight_decay'] == 0
    sched = build_scheduler(opt, cfg)
    lrs = []
    for _ in range(40):
        lrs.append(opt.param_groups[0]['lr'])
        opt.step()
        sched.step()
    assert lrs[0] == lrs[14] == 1e-3
    assert abs(lrs[15] - 1e-4) < 1e-12 and abs(lrs[29] - 1e-4) < 1e-12 and abs(lrs[30] - 1e-5) < 1e-12


def _so3f():
    from pose3d.models.clifford_flow import CliffordFlow
    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False, encoder_type="resnet50",
                         conv_adapter=False, n_time_samples=2, vector_field="so3fourier", cond_tokens="sphere",
                         condition_head="none", so3fourier=dict(c_s=8, c_f=4, lmax=4, hidden=16))
    with torch.no_grad():   # undo the zero-initialised read-out
        model.vector_field.out.weight.normal_(0, 0.5)
        model.vector_field.out.bias.normal_(0, 0.5)
    return model.eval()


def test_so3fourier_divergence_is_finite_at_gimbal_lock():
    """The SO(3)-Fourier flow's likelihood divergence at the identity (Euler-angle gimbal lock), against central
    differences: the Euler-angle Wigner-D gave NaN here and made a SYMSOL log-likelihood NaN."""
    from pose3d.engine.likelihood import _velocity_and_divergence, flow_states

    model = _so3f()
    states = flow_states(model)
    cond = model.condition(torch.rand(1, 3, 224, 224)).expand(2, -1, -1)
    R = torch.stack([torch.eye(3), matrix_flow.exp_map(torch.tensor([0.0, 0.7, 0.0]))])
    s, t = states.from_matrix(R), torch.full((2,), 0.5)
    _, div = _velocity_and_divergence(model, states, s, t, cond)
    assert torch.isfinite(div).all()
    h, fd = 1e-3, torch.zeros(2)
    with torch.no_grad():
        for i in range(3):
            e = torch.zeros(2, 3)
            e[:, i] = h
            fd += (model.velocity(states.step(s, e), t, cond)[:, i] - model.velocity(states.step(s, -e), t, cond)[:, i]) / (2 * h)
    torch.testing.assert_close(div, fd, atol=2e-3, rtol=2e-2)
