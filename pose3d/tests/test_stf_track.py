"""STF track: the tensor algebra (ranks 2..6), the tower, and the two-tower CliffordFlow.

The first half needs only torch. The second half builds CliffordFlow with a GATr vector field and is
skipped when the GATr package is missing (as test_gatr_denoiser.py).
"""

import itertools

import pytest
import torch

from pose3d.models.stf_track import (
    INJECT_MODES, NormGate, StfTower, apply_rotation, contract_down, n_components, rotate_to_frame,
    stf2, stf_basis, stf_feature_blocks, stf_invariants, stf_next, to_components,
)

D = torch.float64
RANKS = range(2, 7)


def random_rotation(n=None, dtype=D, seed=None):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    shape = (3, 3) if n is None else (n, 3, 3)
    q, r = torch.linalg.qr(torch.randn(*shape, generator=gen, dtype=dtype))
    q = q * torch.diagonal(r, dim1=-2, dim2=-1).sign().unsqueeze(-2)
    det = torch.linalg.det(q)
    q[..., :, 0] = q[..., :, 0] * det.sign().unsqueeze(-1)
    return q


def chain(l, batch=4, seed=0):
    """STF_l(STF_{l-1}(...), v_l) from random vectors, plus the vectors."""
    gen = torch.Generator().manual_seed(seed)
    vs = [torch.randn(batch, 3, generator=gen, dtype=D) for _ in range(l)]
    S = vs[0]
    for v in vs[1:]:
        S = stf_next(S, v)
    return S, vs


def max_trace(S, l):
    """Largest |trace| of S over any pair of its l index axes."""
    nb = S.ndim - l
    worst = 0.0
    for a, b in itertools.combinations(range(l), 2):
        worst = max(worst, torch.diagonal(S, dim1=nb + a, dim2=nb + b).sum(-1).abs().max().item())
    return worst


def max_asymmetry(S, l):
    nb = S.ndim - l
    # transpositions of neighbouring axes generate all permutations
    return max((S - S.transpose(nb + a, nb + a + 1)).abs().max().item() for a in range(l - 1))


def test_stf2_is_the_textbook_formula():
    u, v = torch.randn(5, 3, dtype=D), torch.randn(5, 3, dtype=D)
    expect = 0.5 * (u[:, :, None] * v[:, None] + v[:, :, None] * u[:, None]) \
        - (u * v).sum(-1)[:, None, None] * torch.eye(3, dtype=D) / 3
    torch.testing.assert_close(stf2(u, v), expect)
    assert max_trace(stf2(u, v), 2) < 1e-14


def test_stf3_matches_the_cliffordstf_closed_form():
    """CliffordSTF section 3.2: T_ijk = (S_ij v_k + S_ik v_j + S_jk v_i) / 3, A_k = 2/3 S_kl v_l,
    STF_3 = T - 1/5 (delta_ij A_k + delta_ik A_j + delta_jk A_i)."""
    gen = torch.Generator().manual_seed(1)
    S = stf2(torch.randn(6, 3, generator=gen, dtype=D), torch.randn(6, 3, generator=gen, dtype=D))
    v = torch.randn(6, 3, generator=gen, dtype=D)
    T = (torch.einsum("bij,bk->bijk", S, v) + torch.einsum("bik,bj->bijk", S, v)
         + torch.einsum("bjk,bi->bijk", S, v)) / 3
    A = 2.0 / 3.0 * torch.einsum("bkl,bl->bk", S, v)
    eye = torch.eye(3, dtype=D)
    paper = T - (torch.einsum("ij,bk->bijk", eye, A) + torch.einsum("ik,bj->bijk", eye, A)
                 + torch.einsum("jk,bi->bijk", eye, A)) / 5
    torch.testing.assert_close(stf_next(S, v), paper)


@pytest.mark.parametrize("l", RANKS)
def test_stf_next_is_symmetric_and_traceless(l):
    S, _ = chain(l)
    assert S.shape == (4,) + (3,) * l
    assert max_asymmetry(S, l) < 1e-13
    assert max_trace(S, l) < 1e-13          # every index pair
    assert S.abs().max() > 1e-3             # and not trivially zero


@pytest.mark.parametrize("l", RANKS)
def test_stf_next_is_so3_equivariant(l):
    """Rotate the inputs by Q, the output must be the rotated output (Q applied to every index)."""
    Q = random_rotation(4, seed=l)
    gen = torch.Generator().manual_seed(10 + l)
    vs = [torch.randn(4, 3, generator=gen, dtype=D) for _ in range(l)]
    S = vs[0]
    for v in vs[1:]:
        S = stf_next(S, v)
    S_rot = torch.einsum("bij,bj->bi", Q, vs[0])
    for v in vs[1:]:
        S_rot = stf_next(S_rot, torch.einsum("bij,bj->bi", Q, v))
    torch.testing.assert_close(S_rot, apply_rotation(S, Q, l), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("l", RANKS)
def test_vector_parity_is_l(l):
    """v -> -v for every input vector multiplies the rank-l result by (-1)^l (a check of the index count)."""
    S, vs = chain(l)
    flipped = -vs[0]
    for v in vs[1:]:
        flipped = stf_next(flipped, -v)
    torch.testing.assert_close(flipped, (-1) ** l * S, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("l", range(2, 7))
def test_contract_down_keeps_symmetric_traceless_and_is_equivariant(l):
    S, vs = chain(l)
    v = torch.randn(4, 3, dtype=D)
    down = contract_down(S, v)
    r = l - 1
    assert down.shape == (4,) + (3,) * r
    if r >= 2:
        assert max_trace(down, r) < 1e-13      # no re-projection needed: S . v of an STF S is STF
        assert max_asymmetry(down, r) < 1e-13
    Q = random_rotation(4, seed=3)
    torch.testing.assert_close(contract_down(apply_rotation(S, Q, l), torch.einsum("bij,bj->bi", Q, v)),
                               apply_rotation(down, Q, r) if r >= 1 else down, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("l", RANKS)
def test_rotate_to_frame_is_invariant_under_a_joint_left_rotation(l):
    S, _ = chain(l)
    R = random_rotation(4, seed=20)
    g = random_rotation(4, seed=21)
    body = rotate_to_frame(S, R, l)
    body_rot = rotate_to_frame(apply_rotation(S, g, l), g @ R, l)
    torch.testing.assert_close(body_rot, body, atol=1e-12, rtol=1e-12)
    # and it is the inverse of applying R
    torch.testing.assert_close(apply_rotation(body, R, l), S, atol=1e-12, rtol=1e-12)


def test_rotate_to_frame_broadcasts_over_channels():
    S = torch.randn(4, 8, 3, 3, 3, dtype=D)
    R = random_rotation(4, seed=2)
    out = rotate_to_frame(S, R, 3)
    torch.testing.assert_close(out[2, 5], rotate_to_frame(S[2, 5], R[2], 3))


@pytest.mark.parametrize("l", range(1, 7))
def test_stf_basis_has_2l_plus_1_orthonormal_columns(l):
    B = stf_basis(l)
    assert B.shape == (3 ** l, n_components(l)) == (3 ** l, 2 * l + 1)
    torch.testing.assert_close(B.T @ B, torch.eye(2 * l + 1), atol=1e-5, rtol=0)
    if l >= 2:
        S, _ = chain(l)
        c = to_components(S, l)
        assert c.shape == (4, 2 * l + 1)
        # the coordinates carry the whole tensor: same Frobenius norm, exact reconstruction
        torch.testing.assert_close(c.norm(dim=-1), S.flatten(1).norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close((c @ B.T.to(D)).reshape(S.shape), S, atol=1e-5, rtol=1e-5)


def test_stf_basis_columns_are_stf_tensors():
    for l in (2, 4, 6):
        T = stf_basis(l).T.to(D).reshape(2 * l + 1, *([3] * l))
        assert max_trace(T, l) < 1e-6 and max_asymmetry(T, l) < 1e-6


def test_norm_gate_is_invariant():
    torch.manual_seed(0)
    gate = NormGate(8, n_scalars=5).to(D)
    S = torch.randn(3, 8, 3, 3, 3, dtype=D)
    sc = torch.randn(3, 5, dtype=D)
    Q = random_rotation(3, seed=4)
    torch.testing.assert_close(gate(apply_rotation(S, Q, 3), sc), apply_rotation(gate(S, sc), Q, 3),
                               atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lmax", [2, 3, 6])
def test_tower_is_equivariant_and_feedback_is_a_vector(lmax):
    torch.manual_seed(0)
    tower = StfTower(n_vec=12, lmax=lmax, channels=4, n_scalars=6).to(D)
    for p in tower.parameters():
        torch.nn.init.normal_(p, std=0.5)
    V = torch.randn(3, 12, 3, dtype=D)
    sc = torch.randn(3, 6, dtype=D)
    Q = random_rotation(3, seed=5)
    out = tower(V, sc)
    out_rot = tower(torch.einsum("bij,bnj->bni", Q, V), sc)
    assert len(out.stf) == lmax - 1
    for l, (a, b) in zip(range(2, lmax + 1), zip(out.stf, out_rot.stf)):
        assert a.shape == (3, 4) + (3,) * l
        assert max_trace(a, l) < 1e-12 and max_asymmetry(a, l) < 1e-12
        torch.testing.assert_close(b, apply_rotation(a, Q, l), atol=1e-11, rtol=1e-9)
    torch.testing.assert_close(out_rot.feedback, torch.einsum("bij,bnj->bni", Q, out.feedback),
                               atol=1e-11, rtol=1e-9)


def test_tower_gradients_reach_every_parameter():
    torch.manual_seed(0)
    tower = StfTower(n_vec=8, lmax=4, channels=3, n_scalars=4)
    out = tower(torch.randn(2, 8, 3), torch.randn(2, 4))
    (sum(s.pow(2).sum() for s in out.stf) + out.feedback.pow(2).sum()).backward()
    for name, p in tower.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert tower.feedback_scale.grad.abs() > 0


@pytest.mark.parametrize("mode", INJECT_MODES)
def test_invariants_have_the_announced_size_and_are_bounded(mode):
    stf = tuple(torch.randn(5, 8, *([3] * l)) for l in (2, 3, 4))
    R = random_rotation(5, torch.float32, seed=6)
    f = stf_invariants(stf, R, mode)
    assert f.shape == (5, sum(stf_feature_blocks(4, 8, mode)))
    assert torch.isfinite(f).all() and f.abs().max() < 5


def test_gram_is_invariant_under_conjugation_and_frame_under_left_rotation():
    gen = torch.Generator().manual_seed(3)
    stf = tuple(torch.randn(3, 4, *([3] * l), generator=gen, dtype=D) for l in (2, 3, 4))
    R, g = random_rotation(3, seed=7), random_rotation(3, seed=8)
    rot = lambda t: tuple(apply_rotation(s, g, s.ndim - 2) for s in t)

    # conjugation: S -> g.S, R -> g R g^T
    gram = stf_invariants(stf, R, "gram")
    torch.testing.assert_close(stf_invariants(rot(stf), g @ R @ g.transpose(-1, -2), "gram"), gram,
                               atol=1e-6, rtol=1e-6)
    # left rotation: S -> g.S, R -> g R
    frame = stf_invariants(stf, R, "frame")
    torch.testing.assert_close(stf_invariants(rot(stf), g @ R, "frame"), frame, atol=1e-6, rtol=1e-6)
    # and the frame features are NOT invariant under conjugation (that is why "gram" is the default)
    assert (stf_invariants(rot(stf), g @ R @ g.transpose(-1, -2), "frame") - frame).abs().max() > 1e-3
    # the injected scalars do depend on the rotation (degree-l dependence on R)
    assert (stf_invariants(stf, random_rotation(3, seed=9), "gram") - gram).abs().max() > 1e-3


def test_invariants_have_degree_l_dependence_on_the_rotation():
    """Along a one-parameter rotation about z, <S_l, R S_l> for a tensor with all l-components oscillates with
    angular frequency up to l (a multivector cannot produce frequency 3 or more at fixed polynomial degree)."""
    import math
    l = 4
    S = torch.zeros(1, 1, *([3] * l), dtype=D)
    v = torch.tensor([[1.0, 0.5, 0.0]], dtype=D)
    X = v
    for _ in range(l - 1):
        X = stf_next(X, torch.tensor([[1.0, 0.3, 0.6]], dtype=D))
    S[0, 0] = X[0]
    thetas = torch.linspace(0, 2 * math.pi, 64 + 1, dtype=D)[:-1]
    vals = []
    for th in thetas:
        c, s = th.cos(), th.sin()
        R = torch.tensor([[[c, -s, 0], [s, c, 0], [0, 0, 1.0]]], dtype=D)
        vals.append((rotate_to_frame(S, R, l).flatten() * S.flatten()).sum())
    spec = torch.fft.rfft(torch.stack(vals)).abs()
    assert spec[l] > 1e-6 and spec[l + 1:].max() < 1e-10   # frequency l present, nothing above it
