"""dataset modelnet10_points: the point-cloud loader, and the three heads on it. GATr (cond_tokens=points) and
SO(3)-Fourier (cond_tokens=pointsphere) are invariant to turning the cloud and the pose together by ANY rotation
(the physical symmetry with 3D input); DiT (point_input) is not built to be."""

import math

import numpy as np
import pytest
import torch

from pose3d.datasets.modelnet_points import CLASSES, ModelNetPoints, random_rotations, read_off

CUBE = """OFF
8 6 0
-1 -1 -1
1 -1 -1
1 1 -1
-1 1 -1
-1 -1 1
1 -1 1
1 1 1
-1 1 1
4 0 1 2 3
4 4 5 6 7
4 0 1 5 4
4 2 3 7 6
4 1 2 6 5
4 0 3 7 4
"""
WEDGE = """OFF6 2 0
0 0 0
2 0 0
0 1 0
0 0 3
2 0 3
0 1 3
3 0 1 2
3 3 4 5
"""


@pytest.fixture
def root(tmp_path):
    for c, name in enumerate(CLASSES[:3]):
        for split, n in (("train", 3), ("test", 2)):
            d = tmp_path / "ModelNet10" / name / split
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{name}_{i:04d}.off").write_text(CUBE if (c + i) % 2 == 0 else WEDGE)
    return tmp_path


def test_read_off_handles_quads_and_the_header_glitch():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        for text, nf in ((CUBE, 12), (WEDGE, 2)):
            p = os.path.join(d, "m.off")
            open(p, "w").write(text)
            verts, faces = read_off(p)
            assert faces.shape == (nf, 3) and faces.max() < len(verts)


def test_loader(root, tmp_path):
    train = ModelNetPoints(str(root), True, n_points=64, pool=256, cache_dir=str(tmp_path / "c"))
    test = ModelNetPoints(str(root), False, n_points=64, pool=256, cache_dir=str(tmp_path / "c"))
    assert len(train) == 9 and len(test) == 6 * 4
    assert sorted(set(train.cls.tolist())) == [0, 1, 2]
    canon = train.points
    assert canon.norm(dim=-1).max() <= 1 + 1e-5 and abs(canon.norm(dim=-1).max(dim=1).values - 1).max() < 1e-5
    a, b = test[5], test[5]
    torch.testing.assert_close(a["img"], b["img"])                       # fixed test poses and subsets
    R = a["rot"]
    torch.testing.assert_close(R @ R.T, torch.eye(3), atol=1e-5, rtol=0)
    # the cloud is the canonical one turned by the label: x = R x_canonical
    x0 = test.points[5 // 4, test.test_idx[5]]
    torch.testing.assert_close(a["img"], x0 @ R.T)
    assert a["img"].shape == (64, 3) and a["cls"].shape == (1,)


def _cfg_model(args):
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from clifford.algebra.cliffordalgebra import CliffordAlgebra

    torch.manual_seed(0)
    return build_model(parse_args(["--path_to_datasets", "x", "--dataset", "modelnet10_points",
                                   "--no-pretrained_backbone", *args]), CliffordAlgebra((1, 1, 1)))


GATR = ["--vector_field", "gatr", "--cond_tokens", "points", "--pose_tokens", "frame_pin", "--condition_head",
        "none", "--no-gatr_zero_readout", "--gatr_mv_channels", "4", "--gatr_s_channels", "8", "--gatr_blocks", "2",
        "--gatr_heads", "2"]
SO3F = ["--vector_field", "so3fourier", "--cond_tokens", "pointsphere", "--condition_head", "none",
        "--no-so3f_flip_symmetrize", "--so3f_s2_channels", "8", "--so3f_so3_channels", "4", "--so3f_hidden", "16"]
DIT = ["--model", "dit_flow", "--dit_dim", "32"]


def _velocity(model, points, R, t):
    from pose3d.geometry.rotor import matrix_to_rotor
    cond = model.condition(points)
    state = R if type(model).__name__ == "DiTFlow" else matrix_to_rotor(R)
    return model.velocity(state, t, cond)


@pytest.mark.parametrize("args", [GATR, SO3F], ids=["gatr", "so3fourier"])
def test_velocity_invariant_to_turning_cloud_and_pose_together(args):
    if args is GATR:
        from pose3d.models.gatr_denoiser import _ensure_xformers_stub
        _ensure_xformers_stub()
        pytest.importorskip("gatr")
    model = _cfg_model(args).double().eval()
    with torch.no_grad():
        out = model.vector_field.out
        for p in (out.parameters() if hasattr(out, "parameters") else []):
            p.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(3)
    points = torch.randn(2, 40, 3, generator=gen, dtype=torch.float64)
    R = random_rotations(2, gen).double()
    Q = random_rotations(1, gen).double()[0]
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    v = _velocity(model, points, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity(model, points @ Q.T, Q @ R, t), v, atol=1e-8, rtol=1e-6)


def test_dit_is_not_invariant():
    model = _cfg_model(DIT).eval()
    with torch.no_grad():
        for layer in (model.vector_field.final_modulation[-1], model.vector_field.out):
            layer.weight.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(4)
    points = torch.randn(2, 40, 3, generator=gen)
    R, Q = random_rotations(2, gen), random_rotations(1, gen)[0]
    t = torch.tensor([0.3, 0.8])
    v = _velocity(model, points, R, t)
    assert (_velocity(model, points @ Q.T, Q @ R, t) - v).abs().max() > 1e-3


@pytest.mark.parametrize("args", [GATR, SO3F, DIT], ids=["gatr", "so3fourier", "dit"])
def test_trains_and_samples_on_clouds(args):
    if args is GATR:
        from pose3d.models.gatr_denoiser import _ensure_xformers_stub
        _ensure_xformers_stub()
        pytest.importorskip("gatr")
    model = _cfg_model(args).train()
    points = torch.randn(2, 32, 3)
    R = random_rotations(2)
    loss = model.compute_loss(points, R)
    loss.backward()
    assert math.isfinite(loss.item())
    assert model.eval().predict(points, n_samples=2, steps=2).shape == (2, 3, 3)


GATR895 = ["--vector_field", "gatr", "--cond_tokens", "pointcloud", "--pose_tokens", "frame_pin", "--condition_head",
           "gatr", "--no-gatr_zero_readout", "--gatr_mv_channels", "4", "--gatr_s_channels", "8", "--gatr_blocks", "2",
           "--gatr_heads", "2", "--n_cond_mv", "6"]


def test_gatr_895_on_points_is_invariant_and_trains():
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    model = _cfg_model(GATR895).double().eval()
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(5)
    points = torch.randn(2, 30, 3, generator=gen, dtype=torch.float64)
    R, Q = random_rotations(2, gen).double(), random_rotations(1, gen).double()[0]
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    cond = model.condition(points)
    assert cond.shape == (2, 6, 16 + model.condition_head.out_scalars)
    v = _velocity(model, points, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity(model, points @ Q.T, Q @ R, t), v, atol=1e-8, rtol=1e-6)
    model = _cfg_model(GATR895).train()
    loss = model.compute_loss(torch.randn(2, 32, 3), random_rotations(2))
    loss.backward()
    assert math.isfinite(loss.item()) and model.condition_head.query.grad is not None


def test_symsol_points_shapes_and_symmetries():
    from pose3d.datasets.symsol_points import SymsolPoints
    train, test = SymsolPoints(True, n_points=600, views=3), SymsolPoints(False, n_points=600, views=2)
    assert [len(g) for g in train.groups] == [12, 24, 60, 360, 720]
    rng = np.random.default_rng(0)
    for c, G in enumerate(train.groups):
        x = torch.from_numpy(train.sample(c, 600, rng)).float()
        assert x.norm(dim=-1).max() <= 1 + 1e-5
        torch.testing.assert_close(G @ G.transpose(1, 2), torch.eye(3).expand_as(G), atol=1e-5, rtol=0)
        if c < 3:   # polyhedra: every group element maps the vertex set onto itself
            v = torch.from_numpy(train.meshes[train.shapes[c]][0]).float()
            moved = torch.einsum("gij,vj->gvi", G, v)
            d = (moved[:, :, None] - v[None, None]).norm(dim=-1).min(-1).values
            assert d.max() < 1e-5
    item = test[1]
    assert item["img"].shape == (600, 3) and int(item["n_eq"]) == 12
    torch.testing.assert_close(test[1]["img"], item["img"])                    # fixed test items
    eq = item["rots_eq"][:12]
    torch.testing.assert_close(eq[0], item["rot"])
    tr = train[0]
    assert set(tr) == {"img", "rot", "cls_eval"}


def test_symsol_points_fixed_training_set():
    from pose3d.datasets.symsol_points import SymsolPoints
    fixed = SymsolPoints(True, n_points=50, views=3, fixed=True)
    a, b = fixed[4], fixed[4]
    torch.testing.assert_close(a["img"], b["img"])                       # same cloud every epoch
    G = fixed.groups[1]
    rel = a["rot"].T @ b["rot"]                                          # labels differ by a symmetry, at most
    assert (torch.einsum("ij,gij->g", rel, G) > 2.999).any()
    assert not torch.allclose(fixed[3]["img"], fixed[4]["img"])
    free = SymsolPoints(True, n_points=50, views=3)
    assert not torch.allclose(free[4]["img"], free[4]["img"])


def test_gatr_with_pose_harmonics_is_invariant_and_sees_them():
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    model = _cfg_model(GATR + ["--gatr_pose_harmonics", "6"]).double().eval()
    assert model.vector_field.n_harm == 49
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(6)
    points = torch.randn(2, 30, 3, generator=gen, dtype=torch.float64)
    R, Q = random_rotations(2, gen).double(), random_rotations(1, gen).double()[0]
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    v = _velocity(model, points, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity(model, points @ Q.T, Q @ R, t), v, atol=1e-8, rtol=1e-6)
    plain = _cfg_model(GATR).double()
    assert sum(p.numel() for p in model.parameters()) > sum(p.numel() for p in plain.parameters())


def test_gatr_895_with_pose_harmonics_on_points():
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    model = _cfg_model(GATR895 + ["--gatr_pose_harmonics", "6"]).double().eval()
    assert model.vector_field.n_harm == 49
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(7)
    points = torch.randn(2, 30, 3, generator=gen, dtype=torch.float64)
    R, Q = random_rotations(2, gen).double(), random_rotations(1, gen).double()[0]
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    v = _velocity(model, points, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity(model, points @ Q.T, Q @ R, t), v, atol=1e-8, rtol=1e-6)


HYBRID = ["--vector_field", "so3fourier", "--cond_tokens", "pointgatrsphere", "--condition_head", "none",
          "--no-so3f_flip_symmetrize", "--so3f_s2_channels", "8", "--so3f_so3_channels", "4", "--so3f_hidden", "16",
          "--gatr_mv_channels", "4", "--gatr_s_channels", "8", "--gatr_enc_blocks", "1", "--gatr_heads", "2"]


def test_gatr_encoder_fourier_hybrid_is_invariant_and_trains():
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    model = _cfg_model(HYBRID).double().eval()
    with torch.no_grad():
        model.vector_field.out.weight.normal_(0, 0.5)
    gen = torch.Generator().manual_seed(8)
    points = torch.randn(2, 30, 3, generator=gen, dtype=torch.float64)
    R, Q = random_rotations(2, gen).double(), random_rotations(1, gen).double()[0]
    t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    v = _velocity(model, points, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity(model, points @ Q.T, Q @ R, t), v, atol=1e-8, rtol=1e-6)
    model = _cfg_model(HYBRID).train()
    with torch.no_grad():                    # past the zero-initialised read-out, the encoder gets a gradient
        model.vector_field.out.weight.normal_(0, 0.5)
    loss = model.compute_loss(torch.randn(2, 32, 3), random_rotations(2))
    loss.backward()
    assert math.isfinite(loss.item())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.adapter.conv_adapter.encoder.parameters())
