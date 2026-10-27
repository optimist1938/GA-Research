"""Symmetric traceless Cartesian tensors (STF) of rank 2..6 as a second feature track for CliffordFlow.

Why. Under SO(3) every grade of the geometric algebra Cl(3,0) (and of GATr's Cl(3,0,1)) is a scalar
(degree l = 0) or a vector (l = 1): multivectors cannot carry an l >= 2 irreducible representation
(reports/ga_high_degree_irreps.md). A symmetric traceless rank-l tensor ("STF_l": symmetric under
every index permutation, zero trace over every index pair) has 2l + 1 independent components and
transforms as exactly the degree-l irrep, so a second track of such tensors next to the multivectors
gives the flow that missing content. This is the construction of CliffordSTF (Polat et al.,
arXiv 2606.29584, section 3.2), extended from rank 3 to any rank by one recursion.

Everything here is pure PyTorch, closed form, with no Clebsch-Gordan table and no e3nn.

Conventions used in every function below
* A tensor of rank l is stored FULL, as 3**l floats, with shape (*batch, 3, ..., 3) (l trailing axes
  of size 3). Only 2l + 1 of them are independent; `n_components(l)` counts them and `stf_basis(l)`
  gives an orthonormal coordinate system for them. Full storage keeps every formula an index
  expression and is cheap for l <= 6 (729 floats) with a handful of channels.
* "Rotating" a rank-l tensor by a 3 x 3 rotation matrix Q means applying Q to every index:
  (Q . S)_{i1..il} = Q_{i1 j1} ... Q_{il jl} S_{j1..jl}. Vectors are the l = 1 case (Q . v = Q v).
* delta is the 3 x 3 identity matrix, Sym is the average over all permutations of the indices.

Layout of the file: tensor algebra (`stf2`, `stf_next`, `contract_down`, `rotate_to_frame`), the
modules (`NormGate`, `StfTower`), and the helpers that turn the tensors into rotation-invariant
scalars for the vector field (`stf_invariants`, `StfEmbed`).
"""

import functools
import math
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

MAX_RANK = 6   # the largest rank supported (3**6 = 729 floats per channel)
_EPS = 1e-12


def n_components(rank: int) -> int:
    """Independent components of a rank-`rank` STF tensor: 2 * rank + 1 (the dimension of the
    degree-`rank` irrep). Used for logging and for sizing the injected scalars."""
    return 2 * rank + 1


def _tensor_rank(S: torch.Tensor, n_batch: int) -> int:
    """Rank of S when its first `n_batch` axes are batch axes and the rest are index axes."""
    return S.ndim - n_batch


def _delta(like: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------------------------
# Tensor algebra
# --------------------------------------------------------------------------------------------

def stf2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """STF_2(u, v) = (u v^T + v u^T) / 2 - (u . v) delta / 3, the l = 2 part of u (x) v.

    u, v: (*batch, 3).  Returns (*batch, 3, 3), symmetric with zero trace. Together with the scalar
    u . v (l = 0) and the cross product u x v (l = 1) it is the Clebsch-Gordan split 1 x 1 = 0 + 1 + 2.
    """
    outer = u.unsqueeze(-1) * v.unsqueeze(-2)
    dot = (u * v).sum(-1)
    return 0.5 * (outer + outer.transpose(-1, -2)) - dot[..., None, None] * _delta(u) / 3.0


def stf_next(S: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Couple a rank-(l-1) STF tensor S with a vector v into the rank-l STF tensor

        STF_l(S, v) = Sym(S (x) v) - ((l - 1) / (2 l - 1)) * Sym(delta (x) (S . v)),

    where (S . v)_{i1..i_{l-2}} = S_{i1..i_{l-2} k} v_k is S contracted with v on one index, delta (x)
    (S . v) places the identity on two of the l indices, and Sym averages over index permutations.

    Why that coefficient. Sym(S (x) v) is symmetric but has the trace (2 / l) (S . v) over any index
    pair (S is traceless, so only the terms where v sits on a traced index survive). Sym(delta (x) A)
    for a traceless A has the trace (2 (2 l - 1) / (l (l - 1))) A. Subtracting c Sym(delta (x) A) with
    c = (l - 1) / (2 l - 1) therefore removes the trace exactly. For l = 2 this is `stf2`; for l = 3
    it is CliffordSTF's STF_3 (their 1/5 * 2/3 = 2/15 is c / 3 = (2/5) / 3).

    S: (*batch, 3^(l-1)) with l >= 2, v: (*batch, 3). The batch shape is read off v. Returns
    (*batch, 3^l). Equivariant: stf_next(Q.S, Q v) = Q . stf_next(S, v) for every rotation Q.

    Implementation: S is already symmetric, so Sym(S (x) v) is the mean of the l placements of v's
    index (l terms, not l!), and Sym(delta (x) A) the mean over the l (l - 1) / 2 index pairs that
    carry delta (A is symmetric too).
    """
    nb = v.ndim - 1
    rank = _tensor_rank(S, nb) + 1
    if rank < 2 or rank > MAX_RANK:
        raise ValueError(f"stf_next builds ranks 2..{MAX_RANK}, got rank {rank}")

    outer = S.unsqueeze(-1) * v.reshape(v.shape[:-1] + (1,) * (rank - 1) + (3,))
    sym = sum(outer.movedim(-1, nb + j) for j in range(rank)) / rank

    trace_vec = (S * v.reshape(v.shape[:-1] + (1,) * (rank - 2) + (3,))).sum(-1)   # S . v
    with_delta = trace_vec[..., None, None] * _delta(S)                              # (.., 3^(l-2), 3, 3)
    pairs = [(a, b) for a in range(rank) for b in range(a + 1, rank)]
    sym_delta = sum(with_delta.movedim((-2, -1), (nb + a, nb + b)) for a, b in pairs) / len(pairs)

    return sym - (rank - 1) / (2 * rank - 1) * sym_delta


def contract_down(S: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Contract one index of a rank-l tensor S with a vector v: (S . v)_{i1..i_{l-1}} = S_{i1..i_{l-1} k} v_k.

    S: (*batch, 3^l) with l >= 1, v: (*batch, 3). Returns (*batch, 3^(l-1)); for l = 1 a scalar
    (*batch,). This is the inverse coupling of `stf_next` (degree l -> l - 1 inside l (x) 1 = (l-1) + l
    + (l+1)); CliffordSTF's cross-track coupling uses it to feed the grade-1 part of the multivector.

    The result is symmetric AND traceless whenever S is: the trace of S . v over two free indices is a
    trace of S over those indices times v, which is zero. So, unlike for the generic contraction of a
    non-traceless tensor, no re-projection is needed (checked numerically in the tests). It is again
    an STF tensor of rank l - 1, i.e. the degree l - 1 irrep.
    """
    nb = v.ndim - 1
    rank = _tensor_rank(S, nb)
    return (S * v.reshape(v.shape[:-1] + (1,) * (rank - 1) + (3,))).sum(-1)


def apply_rotation(S: torch.Tensor, Q: torch.Tensor, rank: int) -> torch.Tensor:
    """Apply the 3 x 3 matrix Q to every one of the `rank` index axes of S: (Q . S)_{i..} = Q_{ij} .. S_{j..}.

    S: (*batch_S, 3^rank). Q: (*batch_Q, 3, 3) whose batch shape is a prefix of batch_S (e.g. Q (B, 3, 3)
    with S (B, C, 3^rank): the same matrix for all C channels). Q need not be a rotation (the tests
    also use it with Q^T).
    """
    nb = S.ndim - rank
    Q = Q.reshape(Q.shape[:-2] + (1,) * (nb - (Q.ndim - 2)) + (3, 3))
    out = S
    for a in range(rank):
        moved = out.movedim(nb + a, -1)                           # (*batch, rest.., 3)
        flat = moved.reshape(*moved.shape[:nb], -1, 3)            # (*batch, R, 3)
        flat = flat @ Q.transpose(-1, -2)                         # out_i = sum_j Q_ij S_j
        out = flat.reshape(moved.shape).movedim(-1, nb + a)
    return out


def rotate_to_frame(S: torch.Tensor, R: torch.Tensor, rank: int) -> torch.Tensor:
    """Express S in the body frame of the rotation matrix R: apply R^T to every index.

    S: (*batch_S, 3^rank), R: (*batch_R, 3, 3), see `apply_rotation` for the batch rule. Joint
    equivariance (tested): rotating S and R by the same g from the left, S -> g . S and R -> g R,
    leaves the output unchanged, because (g R)^T (g . S) = R^T . S.
    """
    return apply_rotation(S, R.transpose(-1, -2), rank)


@functools.lru_cache(maxsize=None)
def stf_basis(rank: int) -> torch.Tensor:
    """Orthonormal coordinates of the STF_rank subspace: a (3^rank, 2 * rank + 1) float32 matrix B.

    B^T flat(S) are the 2l + 1 independent components of an STF tensor S (flat = reshape to 3^rank);
    flat(S) = B (B^T flat(S)) restores it, and B has orthonormal columns so Frobenius norms are
    preserved. Built, once per rank, from the recursion itself: the tensors
    STF_rank(v, ..., v) for many random unit vectors v span the whole STF_rank space (they are the
    degree-`rank` harmonic polynomials), and their SVD gives an orthonormal basis. The SVD must show
    exactly 2l + 1 non-zero singular values, which also checks the recursion's dimension count.
    """
    if not 1 <= rank <= MAX_RANK:
        raise ValueError(f"rank must be in 1..{MAX_RANK}")
    gen = torch.Generator().manual_seed(rank)
    v = torch.randn(8 * n_components(rank), 3, generator=gen, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    S = v
    for _ in range(rank - 1):
        S = stf_next(S, v)
    _, sing, vh = torch.linalg.svd(S.reshape(S.shape[0], -1), full_matrices=False)
    n = n_components(rank)
    spanned = int((sing > 1e-9 * sing[0]).sum())
    if spanned != n:
        raise RuntimeError(f"STF_{rank} spans {spanned} dimensions, expected {n}")
    return vh[:n].T.contiguous().float()


def to_components(S: torch.Tensor, rank: int) -> torch.Tensor:
    """The 2 * rank + 1 independent components (orthonormal coordinates) of STF tensors S: (*batch, 2 rank + 1)."""
    nb = S.ndim - rank
    return S.reshape(*S.shape[:nb], 3 ** rank) @ stf_basis(rank).to(S)


# --------------------------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------------------------

class NormGate(nn.Module):
    """Equivariant gate of STF channels, as CliffordSTF's norm-gating (Weiler et al., 2018).

    Every channel c of a rank-l tensor S (B, C, 3^l) is multiplied by a scalar
    sigmoid(W [log(1 + ||S_c||_F), LayerNorm(scalars)] + b)_c. The Frobenius norm ||S_c||_F and the
    optional `scalars` (rotation-invariant numbers, e.g. the grade-0 parts of the multivectors) do
    not change when S is rotated, so the gate is invariant and S . gate stays an l-tensor.
    """

    def __init__(self, n_channels: int, n_scalars: int = 0):
        super().__init__()
        self.n_scalars = n_scalars
        self.norm = nn.LayerNorm(n_scalars) if n_scalars else None
        self.linear = nn.Linear(n_channels + n_scalars, n_channels)

    def forward(self, S: torch.Tensor, scalars: torch.Tensor = None) -> torch.Tensor:
        # eps inside the root keeps the gradient of the norm finite at S = 0
        norm = (S.flatten(2).pow(2).sum(-1) + _EPS).sqrt()
        feats = torch.log1p(norm)
        if self.norm is not None:
            feats = torch.cat([feats, self.norm(scalars)], dim=-1)
        gate = torch.sigmoid(self.linear(feats))
        return S * gate.reshape(*gate.shape, *([1] * (S.ndim - 2)))


class StfOutput(NamedTuple):
    """What `StfTower` returns. `stf[i]` is the rank-(i + 2) track, shape (B, channels, 3, ..., 3);
    `feedback` is added to the grade-1 part of the condition multivectors, shape (B, n_vec, 3)."""

    stf: Tuple[torch.Tensor, ...]
    feedback: torch.Tensor


class StfTower(nn.Module):
    """Builds STF channels of rank 2..lmax from vector channels and feeds them back as vectors.

    Input: V (B, n_vec, 3), n_vec vector channels per sample (here the grade-1 parts of the
    condition multivectors), and optionally invariant `scalars` (B, n_scalars) for the gates.

    Upward (vectors -> STF), for each of `channels` channels c:
      1. lmax learned linear mixes of the vector channels give lmax vector sets g_0, g_1, ... (B,
         channels, 3), each squashed by g / (1 + |g|) so its length is below 1 (the l-th STF
         product of vectors scales like |g|^l; without this high ranks would explode or vanish);
      2. S_2 = stf2(g_0, g_1), then S_l = stf_next(S_{l-1}, g_{l-1}) for l = 3..lmax;
      3. after each rank: a learned linear mix over the channel axis (equivariant, since all channels
         of one rank transform alike), then `NormGate`.
    Downward (STF -> vectors, CliffordSTF's cross-track coupling): each rank-l tensor is contracted
    l - 1 times with its own vector set (the same g that built it) down to a vector per channel, a
    learned linear map takes the `channels` vectors to n_vec vectors, the ranks are summed and
    multiplied by the learned scalar `feedback_scale` (initialised small). The caller adds this to
    the grade-1 part of the multivectors, a residual.

    Equivariance: every operation is a rotation-equivariant map of vectors/tensors or acts on
    invariants, so rotating V by Q rotates S_l by Q (as an l-tensor) and the feedback by Q.
    """

    def __init__(self, n_vec: int, lmax: int = 3, channels: int = 8, n_scalars: int = 0,
                 feedback_init: float = 0.1):
        super().__init__()
        if not 2 <= lmax <= MAX_RANK:
            raise ValueError(f"stf_lmax must be in 2..{MAX_RANK}, got {lmax}")
        self.n_vec, self.lmax, self.channels = n_vec, lmax, channels
        self.make_vectors = nn.Linear(n_vec, lmax * channels, bias=False)
        ranks = range(2, lmax + 1)
        self.mix = nn.ModuleDict({str(l): nn.Linear(channels, channels, bias=False) for l in ranks})
        self.gate = nn.ModuleDict({str(l): NormGate(channels, n_scalars) for l in ranks})
        self.to_vectors = nn.ModuleDict({str(l): nn.Linear(channels, n_vec, bias=False) for l in ranks})
        self.feedback_scale = nn.Parameter(torch.tensor(float(feedback_init)))

    @staticmethod
    def _mix_channels(S: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        flat = S.flatten(2).transpose(1, 2)                       # (B, 3^l, C)
        return linear(flat).transpose(1, 2).reshape(S.shape)

    def forward(self, V: torch.Tensor, scalars: torch.Tensor = None) -> StfOutput:
        b = V.shape[0]
        gens = self.make_vectors(V.transpose(1, 2))               # (B, 3, lmax * C)
        gens = gens.transpose(1, 2).reshape(b, self.lmax, self.channels, 3)
        gens = gens / (1.0 + gens.norm(dim=-1, keepdim=True))     # |g| < 1

        stf, vec_of_rank = [], {}
        S = None
        for l in range(2, self.lmax + 1):
            if l == 2:
                S = stf2(gens[:, 0], gens[:, 1])
                vec_of_rank[l] = gens[:, 1]
            else:
                S = stf_next(S, gens[:, l - 1])
                vec_of_rank[l] = gens[:, l - 1]
            S = self.gate[str(l)](self._mix_channels(S, self.mix[str(l)]), scalars)
            stf.append(S)

        feedback = V.new_zeros(b, self.n_vec, 3)
        for l, S in zip(range(2, self.lmax + 1), stf):
            vec = S
            for _ in range(l - 1):
                vec = contract_down(vec, vec_of_rank[l])
            feedback = feedback + self.to_vectors[str(l)](vec.transpose(1, 2)).transpose(1, 2)
        return StfOutput(tuple(stf), self.feedback_scale * feedback)


# --------------------------------------------------------------------------------------------
# From STF tensors to scalars for the vector field
# --------------------------------------------------------------------------------------------

#: How the vector field sees the STF tensors, `FlowConfig.stf_inject`:
#:   "gram"  - ratios <S_c, R . S_c'> / (|S_c| |S_c'|) between channels c, c' of one rank, with R the
#:             current rotation matrix. Invariant when the condition tensors and the rotor are all
#:             conjugated by the same rotation g (S -> g.S, R -> g R g^T), the symmetry of the GATr
#:             vector field itself, so the whole flow keeps exactly the control's equivariance.
#:   "frame" - the components of R^T . S_c (S in the body frame of R) in `stf_basis` coordinates.
#:             Invariant when S and R are rotated from the left (S -> g.S, R -> g R) but NOT under
#:             conjugation, which GATr's own sandwich symmetry uses, so the whole flow is then no
#:             longer equivariant in the control's sense. Absolute (not only relative) information.
INJECT_MODES = ("gram", "frame")


def stf_feature_blocks(lmax: int, channels: int, mode: str) -> Tuple[int, ...]:
    """Sizes of the per-rank blocks of `stf_invariants`: for rank l the channel features
    (channels * channels for "gram", channels * (2l + 1) for "frame") plus `channels` log-norms."""
    if mode not in INJECT_MODES:
        raise ValueError(f"stf_inject must be one of {INJECT_MODES}, got {mode!r}")
    per_rank = (lambda l: channels * channels) if mode == "gram" else (lambda l: channels * n_components(l))
    return tuple(per_rank(l) + channels for l in range(2, lmax + 1))


def stf_invariants(stf: Tuple[torch.Tensor, ...], R: torch.Tensor, mode: str = "gram") -> torch.Tensor:
    """Rotation-invariant scalars (B', n_features) from the STF track and the rotation matrices R (B', 3, 3).

    stf[i]: (B', C, 3^(i+2)), one entry per rank. For each rank l and channel c:
      n_c = |S_c|_F, the unit tensor s_c = S_c / (n_c + 1e-4) (so every feature is bounded however small
      the high-rank tensors are), then the features of `INJECT_MODES[mode]` computed from s_c and R,
      and the log-norm log(n_c + 1e-6) / 10 (the magnitude that the unit tensors lose).
    The ranks are concatenated in the order of `stf_feature_blocks`.
    """
    if mode not in INJECT_MODES:
        raise ValueError(f"stf_inject must be one of {INJECT_MODES}, got {mode!r}")
    feats = []
    for i, S in enumerate(stf):
        l = i + 2
        norm = (S.flatten(2).pow(2).sum(-1) + _EPS).sqrt()                   # (B', C)
        unit = S / (norm + 1e-4).reshape(*norm.shape, *([1] * l))
        body = rotate_to_frame(unit, R, l)                                   # R^T . s
        if mode == "gram":
            blk = torch.einsum("bcx,bdx->bcd", body.flatten(2), unit.flatten(2)).flatten(1)
        else:
            blk = to_components(body, l).flatten(1)
        feats += [blk, 0.1 * torch.log(norm + 1e-6)]
    return torch.cat(feats, dim=-1)


class StfEmbed(nn.Module):
    """Learned embedding of the STF invariants into the scalar channels of the vector field's rotor token.

    One nn.Linear from all `sum(blocks)` invariants (`stf_feature_blocks`) to `out_dim` GATr scalar
    channels followed by SiLU. A linear layer is enough because the invariants are bounded
    (`stf_invariants`) and GATr's own first layer and nonlinearities follow.
    """

    def __init__(self, blocks: Tuple[int, ...], out_dim: int):
        super().__init__()
        self.linear = nn.Linear(sum(blocks), out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.linear(features))


def log_stf_summary(lmax: int, channels: int) -> str:
    """One line for the run log: the ranks, their independent components and the size of the track."""
    per_rank = ", ".join(f"l={l}: {n_components(l)} comps" for l in range(2, lmax + 1))
    total = channels * sum(n_components(l) for l in range(2, lmax + 1))
    full = channels * sum(3 ** l for l in range(2, lmax + 1))
    return (f"STF track: ranks 2..{lmax} x {channels} channels ({per_rank}); "
            f"{total} independent components, stored as {full} floats")
