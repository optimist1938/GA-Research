r"""A small, self-contained Clifford algebra and a set of equivariant layers.

Self-contained on purpose: the sibling pose-estimation project in this repo
depends on an external ``clifford`` package pulled from git, and the PA
experiments should stay runnable without it.  The algebra here is built from a
Cayley table over bitmask-encoded blades, so ``Cl(p, q, r)`` works for any
signature; the PA models use ``Cl(2, 0)``.

Why geometric algebra for a power amplifier
-------------------------------------------
A baseband PA model must satisfy a phase equivariance: rotating the input
constellation rotates the output the same way,

.. math:: x(k) \\mapsto x(k) e^{j\\varphi} \;\\Longrightarrow\; y(k) \\mapsto y(k) e^{j\\varphi}.

Classical memory polynomials get this by *construction* -- every term is
``x * f(|x|)``, with an even-powered envelope and a single unpaired ``x`` -- which
is also why they are restricted to that shape.  In ``Cl(2, 0)`` the I/Q plane is
a genuine vector space, that phase rotation is a rotor sandwich ``v -> R v R~``,
and the layers below are equivariant under it by construction.  The symmetry is
then a property of the architecture rather than of a hand-picked basis, and the
network is free to use products and grades a memory polynomial cannot express.

Equivariance holds because each layer is built only from operations that
commute with (or are covariant under) the rotor action: grade projection,
channel mixing with grade-wise scalar weights, the geometric product itself, and
gating by grade norms, which are invariants.
"""

from __future__ import annotations

import functools

import torch
from torch import nn

__all__ = [
    "CliffordAlgebra",
    "MVLinear",
    "MVGeometricProduct",
    "MVGradeGate",
    "MVInvariantGate",
    "MVLayerNorm",
    "MVBlock",
]


def _reordering_sign(a: int, b: int) -> int:
    """Sign from sorting the concatenated basis-vector list ``a b`` into order."""
    a >>= 1
    total = 0
    while a:
        total += bin(a & b).count("1")
        a >>= 1
    return -1 if (total & 1) else 1


class CliffordAlgebra:
    """``Cl(p, q, r)`` over ``R``, blades indexed by bitmask.

    ``signature`` is ``p`` ones, ``q`` minus-ones and ``r`` zeros.  Blade ``i``
    is the product of the basis vectors whose bit is set in ``i``, so blade 0 is
    the scalar and blade ``2**n - 1`` the pseudoscalar.
    """

    def __init__(self, p: int, q: int = 0, r: int = 0) -> None:
        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2**self.n
        self.metric = [1] * p + [-1] * q + [0] * r
        self.grades = torch.tensor([bin(i).count("1") for i in range(self.dim)], dtype=torch.long)
        self.cayley = self._build_cayley()

    def _build_cayley(self) -> torch.Tensor:
        """``(dim, dim, dim)`` structure constants: ``e_i e_j = sum_k C[i,j,k] e_k``."""
        c = torch.zeros(self.dim, self.dim, self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                sign = _reordering_sign(i, j)
                shared = i & j
                for b in range(self.n):
                    if shared >> b & 1:
                        sign *= self.metric[b]
                if sign:
                    c[i, j, i ^ j] = sign
        return c

    @functools.cached_property
    def grade_masks(self) -> torch.Tensor:
        """``(n + 1, dim)`` 0/1 masks selecting each grade."""
        return torch.stack([(self.grades == g).to(torch.float32) for g in range(self.n + 1)])

    @functools.cached_property
    def _flat_cayley(self) -> torch.Tensor:
        """``(dim * dim, dim)`` view of the structure constants, for matmul."""
        return self.cayley.reshape(self.dim * self.dim, self.dim)

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geometric product over the last axis; broadcasts over leading axes.

        Written as one matmul against the flattened Cayley table rather than a
        three-operand einsum: the outer product ``a_i b_j`` is formed once and
        contracted, which is what makes the trunk cheap enough to train on a CPU.
        """
        a, b = torch.broadcast_tensors(a, b)
        outer = (a.unsqueeze(-1) * b.unsqueeze(-2)).reshape(*a.shape[:-1], self.dim * self.dim)
        return outer @ self._flat_cayley.to(a.dtype).to(a.device)

    def grade_project(self, a: torch.Tensor) -> torch.Tensor:
        """Split the last axis by grade: ``(..., dim) -> (..., n + 1, dim)``."""
        return a.unsqueeze(-2) * self.grade_masks.to(a.dtype).to(a.device)

    def grade_sqnorm(self, a: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean norm of each grade -- a rotor invariant. ``(..., n+1)``."""
        return torch.einsum("...i,gi->...g", a**2, self.grade_masks.to(a.dtype).to(a.device))

    def expand_by_grade(self, per_grade: torch.Tensor) -> torch.Tensor:
        """``(..., n + 1) -> (..., dim)`` by repeating each grade over its blades."""
        return per_grade.index_select(-1, self.grades.to(per_grade.device))

    def embed_complex(self, z: torch.Tensor, e1: int = 1, e2: int = 2) -> torch.Tensor:
        """Complex ``z`` -> multivector with ``Re z`` on blade ``e1``, ``Im z`` on ``e2``.

        With the defaults this is the vector ``Re(z) e_1 + Im(z) e_2`` of
        ``Cl(2, 0)``, for which a phase rotation of ``z`` is the rotor action.
        """
        out = torch.zeros(*z.shape, self.dim, dtype=z.real.dtype, device=z.device)
        out[..., e1] = z.real
        out[..., e2] = z.imag
        return out

    def extract_complex(self, a: torch.Tensor, e1: int = 1, e2: int = 2) -> torch.Tensor:
        """Inverse of :meth:`embed_complex` (drops all other blades)."""
        return torch.complex(a[..., e1], a[..., e2])

    def rotor(self, angle: torch.Tensor, plane: int = 3) -> torch.Tensor:
        """``exp(-angle/2 * e_plane)`` for a unit bivector blade -- used in tests."""
        out = torch.zeros(*angle.shape, self.dim, dtype=angle.dtype, device=angle.device)
        out[..., 0] = torch.cos(angle / 2)
        out[..., plane] = -torch.sin(angle / 2)
        return out

    def sandwich(self, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``r a r~`` with ``r~`` the reverse of ``r``."""
        return self.geometric_product(self.geometric_product(r, a), self.reverse(r))

    def reverse(self, a: torch.Tensor) -> torch.Tensor:
        """Reversion: grade ``g`` picks up ``(-1)^{g(g-1)/2}``."""
        g = self.grades.to(a.device)
        sign = torch.where(((g * (g - 1)) // 2) % 2 == 0, 1.0, -1.0).to(a.dtype)
        return a * sign


class MVLinear(nn.Module):
    """Channel mixing with one scalar weight per (out, in, grade).

    Equivariant because grade projection commutes with the rotor action and the
    weights are scalars.  A bias is only admissible on grade 0, which is
    rotation invariant.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.algebra = algebra
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, algebra.n + 1))
        nn.init.normal_(self.weight, std=1.0 / max(in_channels, 1) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("scalar_blade", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(..., C_in, dim)`` -> ``(..., C_out, dim)``.

        Grade projection only *selects* blades, so instead of building the
        ``(..., C_in, n + 1, dim)`` projection we expand the weights to one entry
        per blade and contract directly.
        """
        w = self.algebra.expand_by_grade(self.weight.to(x.dtype))  # (C_out, C_in, dim)
        out = torch.einsum("...id,oid->...od", x, w)
        if self.bias is not None:
            pad = torch.zeros(self.algebra.dim, dtype=x.dtype, device=x.device)
            pad[0] = 1.0
            out = out + self.bias.to(x.dtype).unsqueeze(-1) * pad
        return out


class MVGeometricProduct(nn.Module):
    """Equivariant bilinear layer: ``MVLinear(x) (x) MVLinear(x)``, then mix.

    Taking two linear maps and multiplying them channel-wise keeps the cost
    linear in the channel count instead of quadratic, which is the standard
    construction in Clifford-group-equivariant networks.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int) -> None:
        super().__init__()
        self.algebra = algebra
        self.left = MVLinear(algebra, channels, channels, bias=False)
        self.right = MVLinear(algebra, channels, channels, bias=False)
        self.out = MVLinear(algebra, channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.algebra.geometric_product(self.left(x), self.right(x)))


class MVGradeGate(nn.Module):
    """Nonlinearity gating each grade by a learned function of its invariant norm."""

    def __init__(self, algebra: CliffordAlgebra, channels: int) -> None:
        super().__init__()
        self.algebra = algebra
        self.scale = nn.Parameter(torch.ones(channels, algebra.n + 1))
        self.shift = nn.Parameter(torch.zeros(channels, algebra.n + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = self.algebra.grade_sqnorm(x)  # (..., C, n+1) -- rotor invariant
        gate = torch.sigmoid(self.scale.to(x.dtype) * inv + self.shift.to(x.dtype))
        return x * self.algebra.expand_by_grade(gate)


class MVLayerNorm(nn.Module):
    """Normalise each grade by its RMS invariant norm across the channel axis."""

    def __init__(self, algebra: CliffordAlgebra, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.algebra = algebra
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(channels, algebra.n + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = self.algebra.grade_sqnorm(x)  # (..., C, n+1)
        rms = torch.sqrt(inv.mean(dim=-2, keepdim=True) + self.eps)  # over channels
        return x * self.algebra.expand_by_grade(self.gain.to(x.dtype) / rms)


class MVInvariantGate(nn.Module):
    """Gate every (channel, grade) by an MLP over *all* the invariants.

    :class:`MVGradeGate` squashes one invariant per (channel, grade), which can
    only produce a saturating function of that one norm.  A power amplifier's
    AM/AM characteristic is a high-order function of the envelope -- the
    reference model spends Chebyshev polynomials up to degree 8 on it -- and it
    mixes the magnitudes of different carriers and delays.  Feeding the whole
    invariant vector through a small MLP gives exactly that, and stays
    equivariant because the MLP sees only rotor invariants and its output
    rescales each grade.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.algebra = algebra
        n_inv = channels * (algebra.n + 1)
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(n_inv, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_inv),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.ones_(self.mlp[-1].bias)  # start as the identity gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = self.algebra.grade_sqnorm(x)  # (..., C, n+1) -- rotor invariant
        # Weights are cast to the input dtype rather than the other way round, so
        # a float32 module still evaluates exactly in a float64 equivariance check
        # (the same convention MVLinear uses).
        h = inv.flatten(-2)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                h = torch.nn.functional.linear(h, layer.weight.to(h.dtype), layer.bias.to(h.dtype))
            else:
                h = layer(h)
        return x * self.algebra.expand_by_grade(h.reshape(*inv.shape))


class MVBlock(nn.Module):
    """Residual geometric-product block: ``x + gate(gp(norm(x)))``.

    ``norm`` is off by default.  Normalising across channels divides out the
    overall envelope scale, which for a PA model is not a nuisance but the
    signal itself -- the AM/AM characteristic *is* the dependence on that scale.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        gate: str = "invariant",
        normalise: bool = False,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.norm = MVLayerNorm(algebra, channels) if normalise else None
        self.gp = MVGeometricProduct(algebra, channels)
        if gate == "invariant":
            self.gate = MVInvariantGate(algebra, channels, hidden)
        elif gate == "grade":
            self.gate = MVGradeGate(algebra, channels)
        else:
            raise ValueError(f"unknown gate {gate!r}; expected 'invariant' or 'grade'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) if self.norm is not None else x
        return x + self.gate(self.gp(h))
