"""PyTorch models for RF PA behavioural modelling.

:class:`MemoryPolynomialPA` is the reference Chebyshev-LUT model as an
``nn.Module``.  It is exactly the structure in ``SimpleNonLinearModel_ML.m``,
but reorganised so it is cheap to evaluate and differentiable end to end:

* The reference convolves every basis term with ``BL`` and then takes a
  coefficient-weighted sum.  Convolution is linear, so summing *first* and
  convolving *once* gives the same signal for ``prod(n+1) * M`` times less work
  (1215 length-221000 convolutions become one).
* The per-part envelope nonlinearity is a separable Chebyshev tensor, so the
  coefficient-weighted sum over multi-indices is a plain tensor contraction
  rather than an explicit loop over 1215 columns.

The model can therefore be fitted two ways, which is the whole point of the
interface: :meth:`fit_least_squares` reproduces the MATLAB solution in closed
form, and the same parameters can then be refined (or trained from scratch) by
gradient descent against any differentiable objective.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import ChebyshevLUTFeatures, PartModel
from .solvers import matlab_pinv_solve, ridge_solve

__all__ = ["chebyshev_features", "MemoryPolynomialPA", "band_limit"]


def chebyshev_features(t: torch.Tensor, n_funcs: int) -> torch.Tensor:
    """``T_0..T_{n_funcs-1}`` evaluated at ``t``; returns ``(n_funcs, *t.shape)``."""
    out = [torch.ones_like(t)]
    if n_funcs > 1:
        out.append(t)
    for _ in range(2, n_funcs):
        out.append(2.0 * t * out[-1] - out[-2])
    return torch.stack(out[:n_funcs])


def band_limit(g: torch.Tensor, bl: torch.Tensor) -> torch.Tensor:
    """``conv(g, bl, 'same')`` for a complex signal and real, odd-length taps."""
    if bl.numel() % 2 == 0:
        raise ValueError("bl must have an odd number of taps")
    pad = bl.numel() // 2
    # conv1d correlates, so the kernel is flipped to get a true convolution.
    kernel = torch.flip(bl, dims=(0,)).to(g.real.dtype).view(1, 1, -1)

    def _f(v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(v.view(1, 1, -1), kernel, padding=pad).view(-1)

    return torch.complex(_f(g.real), _f(g.imag))


class MemoryPolynomialPA(nn.Module):
    """Chebyshev memory-polynomial PA model (the MATLAB reference, in torch).

    Parameters
    ----------
    part_model
        ``(2 + D, M)`` delay structure, rows ``[s; l; u_1..u_D]``.
    n_basis
        ``ModelBasisFuncNum``; the model uses ``n_basis[p] + 1`` basis functions
        along dimension ``p``.
    bl
        Band-limitation FIR taps (odd length, real).
    abs_scale
        ``absScale`` from the reference (0.99).
    quantise
        If ``True``, reproduce MATLAB's ``2^15``-entry magnitude lookup exactly.
        If ``False``, evaluate the Chebyshev argument continuously, which makes
        the model differentiable with respect to the envelope as well as the
        coefficients.  The two differ by < 1e-4 dB in NMSE.
    """

    def __init__(
        self,
        part_model: PartModel,
        n_basis,
        bl: np.ndarray,
        abs_scale: float = 0.99,
        quantise: bool = False,
        mg_max: int = 2**15,
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        super().__init__()
        self.part_model = part_model
        n_basis = [int(n) for n in np.atleast_1d(n_basis)]
        if len(n_basis) == 1:
            n_basis = n_basis * part_model.dim
        if len(n_basis) != part_model.dim:
            raise ValueError(f"n_basis must have {part_model.dim} entries, got {len(n_basis)}")
        self.n_basis = n_basis
        self.n_funcs = [n + 1 for n in n_basis]
        self.abs_scale = abs_scale
        self.quantise = quantise
        self.mg_max = mg_max

        self.register_buffer("bl", torch.as_tensor(np.asarray(bl), dtype=torch.float64))
        self.register_buffer("delays", torch.as_tensor(part_model.matrix, dtype=torch.long))
        # Coefficients as (M, n_funcs[0], ..., n_funcs[D-1]); flattening in
        # Fortran order reproduces the MATLAB column ordering (see lin2tensor).
        self.coef = nn.Parameter(torch.zeros(part_model.n_parts, *self.n_funcs, dtype=dtype))

    # -- coefficient (de)serialisation ----------------------------------------
    @property
    def n_coef(self) -> int:
        return self.coef.numel()

    def flat_coef(self) -> np.ndarray:
        """Coefficients in the MATLAB column order of ``U``."""
        c = self.coef.detach().cpu().numpy()
        return np.concatenate([c[m].reshape(-1, order="F") for m in range(c.shape[0])])

    def load_flat_coef(self, flat: np.ndarray) -> None:
        per = int(np.prod(self.n_funcs))
        blocks = [
            np.asarray(flat[m * per : (m + 1) * per]).reshape(self.n_funcs, order="F")
            for m in range(self.part_model.n_parts)
        ]
        with torch.no_grad():
            self.coef.copy_(torch.as_tensor(np.stack(blocks), dtype=self.coef.dtype))

    # -- forward ---------------------------------------------------------------
    def _cheb_arg(self, mag: torch.Tensor) -> torch.Tensor:
        """Map an envelope magnitude to the Chebyshev argument in ``[-1, 1)``."""
        if self.quantise:
            idx = torch.clamp(torch.round(mag * self.abs_scale * self.mg_max), 0, self.mg_max - 1)
            return 2.0 * ((idx + 0.5) / self.mg_max - 0.5)
        return 2.0 * (mag * self.abs_scale - 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(D, N)`` complex, unit-peak rows.  Returns ``(N,)`` complex."""
        if x.ndim != 2 or x.shape[0] != self.part_model.dim:
            raise ValueError(f"expected x of shape ({self.part_model.dim}, N), got {tuple(x.shape)}")
        g = torch.zeros(x.shape[1], dtype=self.coef.dtype, device=x.device)
        for m in range(self.part_model.n_parts):
            s, l = int(self.delays[0, m]), int(self.delays[1, m])
            # Per-dimension Chebyshev stacks, with l folded into each u_p.
            feats = []
            for p in range(self.part_model.dim):
                u = int(self.delays[2 + p, m])
                mag = torch.roll(x[p], l + u, dims=0).abs()
                feats.append(chebyshev_features(self._cheb_arg(mag), self.n_funcs[p]).to(self.coef.dtype))
            # envelope = sum_n coef[m, n_1..n_D] * prod_p T_{n_p}, contracting one
            # dimension at a time.  feats[p] is (n_funcs[p], N) and must line up
            # with axis 0 and the trailing sample axis of env, so it is reshaped
            # with explicit singleton axes -- plain broadcasting would silently
            # align it with the *last* basis axis instead, which is correct only
            # when D <= 2.
            env = torch.tensordot(self.coef[m], feats[0], dims=([0], [0]))
            for p in range(1, self.part_model.dim):
                shape = (feats[p].shape[0],) + (1,) * (env.ndim - 2) + (feats[p].shape[1],)
                env = (env * feats[p].reshape(shape)).sum(dim=0)
            g = g + torch.roll(x[0], s, dims=0) * env
        return band_limit(g, self.bl)

    # -- closed-form identification -------------------------------------------
    def fit_least_squares(
        self,
        x: np.ndarray,
        d: np.ndarray,
        indices=None,
        solver: str = "matlab_pinv",
        block_size: int = 16384,
        progress=None,
        **solver_kwargs,
    ) -> dict:
        """Identify the coefficients by least squares, as the reference does.

        ``solver`` is ``"matlab_pinv"`` (bit-compatible with the MATLAB script)
        or ``"ridge"``.  Returns the Gram matrix diagnostics.
        """
        feats = ChebyshevLUTFeatures(
            np.asarray(x), self.part_model, self.n_basis, self.bl.cpu().numpy(), self.abs_scale, self.mg_max
        )
        rx, ry = feats.gram(np.asarray(d), indices=indices, block_size=block_size, progress=progress)
        if solver == "matlab_pinv":
            coef = matlab_pinv_solve(rx, ry, **solver_kwargs)
        elif solver == "ridge":
            coef = ridge_solve(rx, ry, **solver_kwargs)
        else:
            raise ValueError(f"unknown solver {solver!r}; expected 'matlab_pinv' or 'ridge'")
        self.load_flat_coef(coef)
        return {"n_coef": feats.n_coef, "rx": rx, "ry": ry, "features": feats}
