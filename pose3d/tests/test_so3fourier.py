"""vector_field='so3fourier' + cond_tokens='sphere': the SO(3) Fourier condition and the velocity field are exactly
equivariant to 90 degree turns and left-right flips of the backbone map; the canonicalize=False twin is not."""

import math

import pytest
import torch

e3nn = pytest.importorskip("e3nn")
pytest.importorskip("healpy")
from e3nn import o3  # noqa: E402

from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix  # noqa: E402
from pose3d.models.so3fourier_head import (  # noqa: E402
    SO3FourierVelocity, SphereConditionHead, hemisphere_grid, wigner_blocks, wigner_from_harmonics)

R_Z_MINUS_90 = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
FLIP = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
S = torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)


@pytest.fixture(autouse=True)
def _float64_default():
    # e3nn's reference D_from_matrix is only ~1e-6 accurate under a float32 default dtype.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _rot90(fmap):
    return torch.rot90(fmap, 1, dims=(2, 3))


def _flip(fmap):
    return torch.flip(fmap, dims=(3,))


def _left(blocks_D, c, c_f):
    """Apply D^l(G) block-wise to coefficients c (B, c_f, N)."""
    out, start = [], 0
    for l, D in enumerate(blocks_D):
        n = 2 * l + 1
        cl = c[:, :, start:start + n * n].view(c.shape[0], c_f, n, n)
        out.append(torch.einsum("bmk,bykn->bymn", D.expand(c.shape[0], -1, -1), cl).flatten(2))
        start += n * n
    return torch.cat(out, -1)


def _head(c_in=32, c_s=8, c_f=4, lmax=6):
    torch.manual_seed(0)
    head = SphereConditionHead(c_in=c_in, c_s=c_s, c_f=c_f, lmax=lmax).double()
    with torch.no_grad():
        head.radial_gain.normal_(1, 0.3)
        head.l_gain.normal_(1, 0.3)
    return head


def _readout(c_f=4, lmax=6, hidden=16, flip_symmetrize=True, canonicalize=True):
    torch.manual_seed(1)
    vf = SO3FourierVelocity(c_f=c_f, lmax=lmax, hidden=hidden, flip_symmetrize=flip_symmetrize,
                            canonicalize=canonicalize).double()
    with torch.no_grad():
        vf.out.weight.normal_(0, 0.5)
        vf.out.bias.normal_(0, 0.5)
    return vf


def test_hemisphere_grid_is_d4_symmetric():
    pts = hemisphere_grid(4).double()
    for G in (R_Z_MINUS_90, FLIP):
        moved = pts @ G.T
        assert (torch.cdist(moved, pts).min(dim=1).values < 1e-6).all()


@pytest.mark.parametrize("transform,G", [(_rot90, R_Z_MINUS_90), (_flip, FLIP)])
def test_s2_coeffs_rotate_with_the_map(transform, G):
    head = _head()
    fmap = torch.randn(2, 32, 7, 7, dtype=torch.float64)
    irreps = o3.Irreps([(1, (l, (-1) ** l)) for l in range(head.lmax + 1)])   # natural parity: D(F) defined
    D = irreps.D_from_matrix(G)
    psi, psi_t = head.s2_coeffs(fmap), head.s2_coeffs(transform(fmap))
    assert psi.abs().max() > 1e-3
    torch.testing.assert_close(psi_t, psi @ D.T, atol=1e-10, rtol=1e-8)


def test_so3_coeffs_transform_by_the_left_action():
    head = _head()
    fmap = torch.randn(2, 32, 7, 7, dtype=torch.float64)
    D = wigner_blocks(R_Z_MINUS_90[None], head.lmax)
    torch.testing.assert_close(head(_rot90(fmap)), _left(D, head(fmap), head.c_f), atol=1e-10, rtol=1e-8)


def test_readout_invariant_under_any_rotation():
    head, vf = _head(), _readout()
    c = head(torch.randn(3, 32, 7, 7, dtype=torch.float64))
    R, G = o3.rand_matrix(3).double(), o3.rand_matrix(3).double()
    t = torch.rand(3, dtype=torch.float64)
    cG = torch.cat([_left([Dl[i:i + 1] for Dl in wigner_blocks(G, head.lmax)], c[i:i + 1], head.c_f)
                    for i in range(3)])
    v = vf(R, t, c)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(vf(G @ R, t, cG), v, atol=1e-10, rtol=1e-8)


def test_wigner_matches_e3nn_and_follows_the_module_device():
    vf = _readout()
    R = o3.rand_matrix(4).double()
    ref = [o3.wigner_D(l, *o3.matrix_to_angles(R)) for l in range(vf.lmax + 1)]
    for D, D_ref in zip(wigner_blocks(R, vf.lmax, vf.harmonic_basis()), ref):
        torch.testing.assert_close(D, D_ref, atol=1e-10, rtol=1e-8)
    # The basis moves with the module (e3nn's wigner_D builds its constants on the CPU and broke on cuda).
    vf = vf.to("meta")
    X, pinvs = vf.harmonic_basis()
    assert all(D.device.type == "meta" for D in wigner_from_harmonics(torch.eye(3, device="meta")[None], X, pinvs))


def test_wigner_derivative_is_finite_at_gimbal_lock():
    """d D(R exp(e)) / de at poses where e3nn's Y-X-Y Euler angles are singular (beta = 0: rotations about y,
    and the identity), against central differences: the flow likelihood differentiates through here."""
    vf = _readout()
    basis = vf.harmonic_basis()
    for R in (torch.eye(3, dtype=torch.float64), o3.matrix_y(torch.tensor(0.7, dtype=torch.float64))):
        e = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        K = torch.zeros(3, 3, dtype=torch.float64)
        K = K.index_put((torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([1, 2, 0, 2, 0, 1])),
                        torch.stack([-e[2], e[1], e[2], -e[0], -e[1], e[0]]))
        D = wigner_blocks((R @ (torch.eye(3, dtype=torch.float64) + K))[None], vf.lmax, basis)[vf.lmax][0]
        g = torch.autograd.grad(D.sum(), e)[0]
        assert torch.isfinite(g).all()
        h, fd = 1e-6, torch.zeros(3, dtype=torch.float64)
        for i in range(3):
            w = torch.zeros(3, dtype=torch.float64); w[i] = h
            Rp, Rm = R @ torch.linalg.matrix_exp(_hat(w)), R @ torch.linalg.matrix_exp(_hat(-w))
            fd[i] = (wigner_blocks(Rp[None], vf.lmax, basis)[vf.lmax].sum()
                     - wigner_blocks(Rm[None], vf.lmax, basis)[vf.lmax].sum()) / (2 * h)
        torch.testing.assert_close(g, fd, atol=1e-6, rtol=1e-5)


def _hat(w):
    return torch.stack([torch.stack([w.new_zeros(()), -w[2], w[1]]), torch.stack([w[2], w.new_zeros(()), -w[0]]),
                        torch.stack([-w[1], w[0], w.new_zeros(())])])


def _flow(dtype=torch.float64, canonicalize=True, flip_symmetrize=True):
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="so3fourier", cond_tokens="sphere", condition_head="none",
                         so3fourier=dict(c_s=8, c_f=4, lmax=4, hidden=16, canonicalize=canonicalize,
                                         flip_symmetrize=flip_symmetrize)).to(dtype)
    with torch.no_grad():   # undo the zero-initialised read-out
        model.vector_field.out.weight.normal_(0, 0.5)
        model.vector_field.out.bias.normal_(0, 0.5)
    return model.eval()


def _velocity_from_map(model, fmap, R, t):
    return model.velocity(matrix_to_rotor(R), t, model.condition_head(model.adapter.conv_adapter(fmap)))


def test_velocity_invariant_to_turning_image_and_pose_together():
    model = _flow()
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity_from_map(model, _rot90(fmap), R_Z_MINUS_90 @ R, t), v,
                               atol=1e-8, rtol=1e-6)


def test_velocity_flips_with_a_mirrored_image():
    model = _flow()
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.2, 0.9], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, R, t)
    torch.testing.assert_close(_velocity_from_map(model, _flip(fmap), FLIP @ R @ FLIP, t), S * v,
                               atol=1e-8, rtol=1e-6)


def test_flip_breaks_without_symmetrisation_but_roll_holds():
    model = _flow(flip_symmetrize=False)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.4, 0.6], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, R, t)
    torch.testing.assert_close(_velocity_from_map(model, _rot90(fmap), R_Z_MINUS_90 @ R, t), v,
                               atol=1e-8, rtol=1e-6)
    assert (_velocity_from_map(model, _flip(fmap), FLIP @ R @ FLIP, t) - S * v).abs().max() > 1e-3


def test_twin_is_not_equivariant():
    model = _flow(canonicalize=False)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.4, 0.6], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, R, t)
    assert (_velocity_from_map(model, _rot90(fmap), R_Z_MINUS_90 @ R, t) - v).abs().max() > 1e-3


def test_zero_init_readout_gets_a_gradient():
    torch.set_default_dtype(torch.float32)   # a float32 model, as in training (the fixture restores it)
    model = _flow(torch.float32)
    with torch.no_grad():
        model.vector_field.out.weight.zero_()
        model.vector_field.out.bias.zero_()
    model.train()
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert model.vector_field.out.weight.grad.abs().max() > 1e-6


@pytest.mark.parametrize("canonicalize", [True, False])
def test_so3fourier_flow_trains_and_samples(canonicalize):
    torch.set_default_dtype(torch.float32)   # a float32 model, as in training (the fixture restores it)
    model = _flow(torch.float32, canonicalize=canonicalize).train()
    img = torch.randn(2, 3, 224, 224)
    loss = model.compute_loss(img, rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)


def test_so3fourier_needs_sphere_tokens():
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    with pytest.raises(ValueError):
        CliffordFlow(CliffordAlgebra((1, 1, 1)), pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="so3fourier", cond_tokens="pooled", condition_head="none")


@pytest.mark.parametrize("c_s,c_f,hidden,target", [(32, 8, 44, 246_499), (64, 16, 124, 1_116_835)])
def test_param_counts_and_twin_parity(c_s, c_f, hidden, target):
    n = lambda m: sum(p.numel() for p in m.parameters())
    head = SphereConditionHead(c_in=2048, c_s=c_s, c_f=c_f, lmax=6)
    eq = SO3FourierVelocity(c_f=c_f, lmax=6, hidden=hidden)
    twin = SO3FourierVelocity(c_f=c_f, lmax=6, hidden=hidden, canonicalize=False)
    assert abs(n(head) + n(eq) - target) <= 0.03 * target
    assert abs(n(twin) - n(eq)) <= 0.005 * (n(head) + n(eq))


def _hybrid_flow(dtype=torch.float64):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow

    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="so3fourier", cond_tokens="spheregatr", condition_head="none",
                         so3fourier=dict(c_s=8, c_f=4, lmax=4, hidden=16),
                         gatr_encoder=dict(num_blocks=1, mv_channels=4, s_channels=8, num_heads=2)).to(dtype)
    with torch.no_grad():
        model.vector_field.out.weight.normal_(0, 0.5)
        model.vector_field.out.bias.normal_(0, 0.5)
        model.adapter.conv_adapter.radial_gain.normal_(1, 0.3)
    return model.eval()


def test_sphere_gatr_hybrid_turns_and_flips_with_the_map():
    model = _hybrid_flow()
    torch.manual_seed(3)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, R, t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(_velocity_from_map(model, _rot90(fmap), R_Z_MINUS_90 @ R, t), v, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(_velocity_from_map(model, _flip(fmap), FLIP @ R @ FLIP, t), S * v, atol=1e-8, rtol=1e-6)


def test_sphere_gatr_hybrid_trains_and_samples():
    torch.set_default_dtype(torch.float32)
    model = _hybrid_flow(torch.float32).train()
    img = torch.randn(2, 3, 224, 224)
    loss = model.compute_loss(img, rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.adapter.conv_adapter.encoder.parameters())
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)
