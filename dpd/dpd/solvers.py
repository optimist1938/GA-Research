"""Least-squares solvers for the linear-in-parameters PA model.

The reference code solves the normal equations with ``pinv(U'U) * U'd``.  That
is convenient but numerically poor: on GeoData_TB the Gram matrix has condition
number ~8e20, i.e. ``U`` itself is conditioned at ~3e10 and squaring it destroys
most of the available precision.  The pseudo-inverse hides the damage by
truncating the tiny singular values, which is why the *prediction* is stable
even though the coefficient vector is not (it is only defined up to the
numerical null space).

:func:`matlab_pinv_solve` reproduces the reference exactly, including MATLAB's
tolerance rule.  :func:`ridge_solve` is the better-behaved alternative used for
the PyTorch experiments.
"""

from __future__ import annotations

import numpy as np

__all__ = ["matlab_pinv_solve", "ridge_solve", "effective_rank", "MATLAB_RCOND"]


def _matlab_rcond(shape) -> float:
    """MATLAB ``pinv`` default: ``tol = max(size(A)) * eps(norm(A))``.

    Expressed relative to the largest singular value this is
    ``max(m, n) * eps``.  NumPy's default (``rcond=1e-15``) is ~270x smaller for
    a 1215x1215 matrix, keeps 15 extra near-null singular values, and yields a
    minimum-norm solution ~13x larger in norm -- same prediction, different
    coefficients.  Matching the rule is what makes the two ports agree.
    """
    return max(shape) * np.finfo(np.float64).eps


MATLAB_RCOND = _matlab_rcond


def matlab_pinv_solve(rx: np.ndarray, ry: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """``pinv(RX) * RY``, or ``pinv(RX, tol) * RY`` when ``tol > 0``.

    ``tol`` follows the reference convention: ``0`` means "MATLAB default".
    """
    rcond = _matlab_rcond(rx.shape) if tol == 0 else tol / np.linalg.norm(rx, 2)
    return np.linalg.pinv(rx, rcond=rcond, hermitian=True) @ ry


def ridge_solve(rx: np.ndarray, ry: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    """``(RX + alpha * tr(RX)/n * I)^-1 RY`` -- Tikhonov on the normal equations.

    ``alpha`` is relative to the mean diagonal of ``RX`` so it is scale-free.
    The default was chosen on the D = 3 GeoData_TB fit, where held-out NMSE is
    flat to ~0.1 dB over ``alpha`` in 1e-7..1e-4 and best at 1e-6; the
    unregularised pseudo-inverse is 0.9 dB worse out of sample there and
    produces a coefficient vector two orders of magnitude larger in norm.
    """
    n = rx.shape[0]
    shift = alpha * float(np.real(np.trace(rx))) / n
    return np.linalg.solve(rx + shift * np.eye(n, dtype=rx.dtype), ry)


def effective_rank(rx: np.ndarray, rcond: float | None = None) -> tuple[int, float]:
    """``(numerical rank, condition number)`` of a Hermitian Gram matrix."""
    s = np.linalg.svd(rx, compute_uv=False)
    if rcond is None:
        rcond = _matlab_rcond(rx.shape)
    return int((s > rcond * s[0]).sum()), float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
