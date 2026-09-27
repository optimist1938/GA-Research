r"""Regressor construction for the reference (Chebyshev-LUT) PA model.

Model, following ``SimpleNonLinearModel_ML.m`` and the supplied description:

.. math::

    y(k) \;=\; \\sum_{m} \\sum_{n_1 \\dots n_D} c_{m,\\mathbf{n}} \;
        \\Big( \\mathrm{BL} * \\big[\\, x_1(k - s_m)
        \\prod_{p=1}^{D} T_{n_p}\\!\\big( |x_p(k - l_m - u_{p,m})| \\big) \\,\\big] \\Big)(k)

``m`` indexes the columns of the *part model* -- a ``(2 + D, M)`` integer matrix
whose rows are ``[s; l; u_1; ...; u_D]``.  ``T_n`` are Chebyshev polynomials of
the first kind, evaluated through a ``2^15``-entry lookup table indexed by the
quantised envelope magnitude, exactly as MATLAB does it.

The reference implementation materialises the full regressor matrix ``U``
(``N x M*prod(n+1)``) before forming ``U'U``.  At D = 3 that is tens of
gigabytes, so :class:`ChebyshevLUTFeatures` streams over time instead and
accumulates the Gram matrix in blocks; the result is numerically identical up to
summation order.

One algebraic simplification is used.  MATLAB computes the basis product from
``|x_p(k - u_p)|`` and *then* delays the product by ``l``.  Delay is linear and
commutes with the pointwise product, so we fold ``l`` into each magnitude delay
(``u_p -> l + u_p``) and skip an array copy per basis term.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .matlab import conv_same, gen_spl_chebyshev, tensor_order_table

__all__ = ["PartModel", "DEFAULT_PART_MODEL", "ChebyshevLUTFeatures"]

# The part model shipped in NonLinearProblemSimple.m (rows s, l, u, v).  Row 5
# is *our* extension for the third band: GeoData_TB is tri-band but the archive
# only ships a 2-dimensional structure, so a third row of delays has to be
# chosen.  It is a copy of row 4 shifted by one tap, which keeps the C-band
# delays in the same range as the others without duplicating row 4 exactly.
DEFAULT_PART_MODEL = np.array(
    [
        [0, 2, -1, 1, 2, 0, 8, 3, 0, 1, 1, 10, 0, -2, 2],  # s  - delay of the linear factor
        [1, 2, 0, 1, 1, 0, 0, 1, 1, 2, -1, 0, 2, 0, 2],    # l  - common delay of the envelope term
        [-1, -2, -1, 0, 1, 1, 4, 2, 0, -2, 0, 0, 0, -2, -1],  # u1 - extra delay on |x_1|
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],     # u2 - extra delay on |x_2|
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],     # u3 - extra delay on |x_3|  (added here)
    ],
    dtype=np.int64,
)


@dataclasses.dataclass(frozen=True)
class PartModel:
    """The ``(2 + D, M)`` delay structure of the model."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] < 3:
            raise ValueError(f"part model must be (2 + D, M) with D >= 1, got {self.matrix.shape}")

    @property
    def dim(self) -> int:
        return self.matrix.shape[0] - 2

    @property
    def n_parts(self) -> int:
        return self.matrix.shape[1]

    @property
    def max_abs_delay(self) -> int:
        return int(np.abs(self.matrix).max())

    @classmethod
    def default(cls, dim: int) -> "PartModel":
        """The shipped structure truncated to ``dim`` nonlinearity dimensions."""
        if not 1 <= dim <= DEFAULT_PART_MODEL.shape[0] - 2:
            raise ValueError(f"dim must be in 1..{DEFAULT_PART_MODEL.shape[0] - 2}, got {dim}")
        return cls(DEFAULT_PART_MODEL[: 2 + dim].copy())

    @classmethod
    def grid(cls, dim: int, s_taps, l_taps=(0,), u_taps=(0,)) -> "PartModel":
        """A regular delay grid, as an alternative to the hand-tuned default."""
        import itertools

        cols = [
            [s, l, *u]
            for s, l, u in itertools.product(s_taps, l_taps, itertools.product(u_taps, repeat=dim))
        ]
        return cls(np.array(cols, dtype=np.int64).T)


class ChebyshevLUTFeatures:
    """Streaming builder for the regressor matrix ``U``.

    Parameters
    ----------
    x
        ``(D, N)`` complex envelopes, each row scaled to unit peak magnitude.
    part_model
        Delay structure; ``part_model.dim`` must equal ``D``.
    n_basis
        Per-dimension Chebyshev order ``ModelBasisFuncNum``; the number of basis
        functions per dimension is ``n_basis[p] + 1``.
    bl
        Band-limitation FIR taps (odd length).
    abs_scale
        ``absScale`` -- keeps the LUT index inside the table (MATLAB uses 0.99).
    """

    def __init__(
        self,
        x: np.ndarray,
        part_model: PartModel,
        n_basis,
        bl: np.ndarray,
        abs_scale: float = 0.99,
        mg_max: int = 2**15,
    ) -> None:
        x = np.asarray(x, dtype=np.complex128)
        if x.ndim != 2:
            raise ValueError(f"x must be (D, N), got shape {x.shape}")
        if x.shape[0] != part_model.dim:
            raise ValueError(f"part model has dim {part_model.dim} but x has {x.shape[0]} rows")
        peak = np.abs(x).max(axis=1)
        if np.any(peak > 1.0 + 1e-9):
            raise ValueError(f"rows of x must have unit peak magnitude, got max |x| = {peak}")
        if len(bl) % 2 == 0:
            raise ValueError("bl must have an odd number of taps for conv(..., 'same') parity")

        self.x = x
        self.part_model = part_model
        self.n_basis = [int(n) for n in np.atleast_1d(n_basis)]
        if len(self.n_basis) == 1:
            self.n_basis = self.n_basis * part_model.dim
        if len(self.n_basis) != part_model.dim:
            raise ValueError(f"n_basis must have {part_model.dim} entries, got {len(self.n_basis)}")
        self.bl = np.asarray(bl, dtype=np.float64)
        self.halo = len(self.bl) // 2
        self.abs_scale = abs_scale
        self.mg_max = mg_max

        # splNum1D = ModelBasisFuncNum + 1
        self.n_funcs = [n + 1 for n in self.n_basis]
        self.tables = [gen_spl_chebyshev(nf, mg_max) for nf in self.n_funcs]
        self.orders = tensor_order_table(self.n_funcs)  # lin2tensor column ordering

        self._lut_index, self._linear = self._precompute_delays()

    # -- shape -----------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.x.shape[1]

    @property
    def n_coef_per_part(self) -> int:
        return int(np.prod(self.n_funcs))

    @property
    def n_coef(self) -> int:
        return self.n_coef_per_part * self.part_model.n_parts

    # -- internals -------------------------------------------------------------
    def _precompute_delays(self):
        """Quantised LUT indices and delayed linear factors, one set per part."""
        pm = self.part_model.matrix
        lut_index, linear = [], []
        for m in range(self.part_model.n_parts):
            s, l = int(pm[0, m]), int(pm[1, m])
            idx = []
            for p in range(self.part_model.dim):
                u = int(pm[2 + p, m])
                mag = np.abs(np.roll(self.x[p], l + u))  # folded delay, see module docstring
                q = np.round(mag * self.abs_scale * self.mg_max).astype(np.int64)
                idx.append(np.clip(q, 0, self.mg_max - 1))
            lut_index.append(idx)
            linear.append(np.roll(self.x[0], s))
        return lut_index, linear

    def _block(self, start: int, stop: int, out: np.ndarray) -> None:
        """Fill ``out[: stop - start, :]`` with the regressor rows ``[start, stop)``."""
        lo = max(0, start - self.halo)
        hi = min(self.n_samples, stop + self.halo)
        left, right = start - lo, hi - stop
        col = 0
        for m in range(self.part_model.n_parts):
            lin = self._linear[m][lo:hi]
            # (n_funcs[p], block) slices of each 1-D basis, gathered once per part
            sliced = [self.tables[p][:, self._lut_index[m][p][lo:hi]] for p in range(self.part_model.dim)]
            for order in self.orders:
                v = lin
                for p in range(self.part_model.dim):
                    v = v * sliced[p][order[p]]
                conv = conv_same(v, self.bl)
                out[: stop - start, col] = conv[left : len(conv) - right] if right else conv[left:]
                col += 1

    def blocks(self, indices: np.ndarray | slice | None = None, block_size: int = 16384):
        """Yield ``(row_indices, U_block)`` covering ``indices``.

        ``indices`` selects which time samples are wanted; blocks are cut on the
        *underlying* time axis so the band-limiting convolution keeps its halo.
        """
        if indices is None:
            indices = slice(0, self.n_samples)
        if isinstance(indices, slice):
            lo, hi, _ = indices.indices(self.n_samples)
            ranges = [(a, min(a + block_size, hi)) for a in range(lo, hi, block_size)]
            select = None
        else:
            indices = np.asarray(indices)
            lo, hi = int(indices.min()), int(indices.max()) + 1
            ranges = [(a, min(a + block_size, hi)) for a in range(lo, hi, block_size)]
            select = indices

        buf = np.empty((block_size, self.n_coef), dtype=np.complex128)
        for a, b in ranges:
            self._block(a, b, buf)
            rows = np.arange(a, b)
            block = buf[: b - a]
            if select is not None:
                keep = np.isin(rows, select)
                rows, block = rows[keep], block[keep]
                if rows.size == 0:
                    continue
            yield rows, block

    def gram(self, d: np.ndarray, indices=None, block_size: int = 16384, progress=None):
        """Accumulate ``(U^H U, U^H d)`` over ``indices`` without materialising ``U``."""
        rx = np.zeros((self.n_coef, self.n_coef), dtype=np.complex128)
        ry = np.zeros(self.n_coef, dtype=np.complex128)
        for rows, block in self.blocks(indices, block_size):
            rx += block.conj().T @ block
            ry += block.conj().T @ d[rows]
            if progress is not None:
                progress(rows[-1] + 1)
        return rx, ry

    def predict(self, coef: np.ndarray, indices=None, block_size: int = 16384):
        """Evaluate ``U @ coef``; returns ``(row_indices, y)``."""
        out_rows, out_y = [], []
        for rows, block in self.blocks(indices, block_size):
            out_rows.append(rows)
            out_y.append(block @ coef)
        return np.concatenate(out_rows), np.concatenate(out_y)

    def matrix(self, indices=None, block_size: int = 16384) -> tuple[np.ndarray, np.ndarray]:
        """Materialise ``U`` in full -- only for small configurations."""
        rows, blocks = [], []
        for r, b in self.blocks(indices, block_size):
            rows.append(r)
            blocks.append(b.copy())
        return np.concatenate(rows), np.concatenate(blocks, axis=0)
