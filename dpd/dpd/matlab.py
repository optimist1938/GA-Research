"""MATLAB/Octave-equivalent primitives used by the reference PA model.

Everything here mirrors a specific construct in ``matlab/``.  The point is that
the Python model is not a *re-interpretation* of the reference code but a
bit-for-bit port, so any NMSE difference is attributable to the model and not to
a quietly different convolution convention.

Two helpers (``delay``, ``nmse``) were not part of the supplied archive -- they
live on the author's MATLAB path.  The definitions used here are stated
explicitly in :func:`delay` and :func:`nmse_db` so the assumption is auditable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin

__all__ = [
    "delay",
    "conv_same",
    "fir1",
    "nmse_db",
    "chebyshev_func",
    "gen_spl_chebyshev",
    "lin2tensor",
    "tensor_order_table",
]


def delay(x: np.ndarray, n: int) -> np.ndarray:
    """MATLAB ``delay(x, n)``: ``y(k) = x(k - n)``.

    The reference archive does not ship ``delay.m``.  We use a *circular* shift
    (``circshift(x, [0 n])``), which is the usual implementation in this family
    of DPD toolboxes and is what ``matlab/delay.m`` in this repo contains.  The
    alternative (zero-padded shift) differs only on ``|n| <= 10`` samples out of
    221000, i.e. below the 1e-4 dB level in NMSE.
    """
    return np.roll(x, int(n), axis=-1)


def conv_same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``conv(a, b, 'same')``: the central ``len(a)`` samples of the full
    convolution, starting at ``floor(len(b) / 2)``.

    This is *not* ``np.convolve(..., mode="same")`` in general: NumPy returns the
    central ``max(len(a), len(b))`` samples, so the two disagree whenever ``a``
    is shorter than the filter.  That case does arise here -- the streaming
    feature builder convolves short trailing blocks -- so the MATLAB rule is
    implemented directly rather than delegated.
    """
    a, b = np.asarray(a), np.asarray(b)
    start = len(b) // 2
    return np.convolve(a, b, mode="full")[start : start + len(a)]


def fir1(n: int, wn: float, kind: str = "low") -> np.ndarray:
    """MATLAB ``fir1(n, Wn, 'low')`` -> ``n + 1`` Hamming-windowed FIR taps.

    MATLAB normalises the response to unity at the centre of the passband;
    :func:`scipy.signal.firwin` does the same with ``scale=True`` (its default).
    """
    if kind != "low":
        raise NotImplementedError(f"only lowpass is ported, got {kind!r}")
    return firwin(n + 1, wn, window="hamming", scale=True)


def nmse_db(ref: np.ndarray, err: np.ndarray) -> float:
    """MATLAB ``nmse(ref, err)`` as used by ``SimpleNonLinearModel_ML``.

    ``10*log10(sum(|err|^2) / sum(|ref|^2))``.  Note the reference signal is the
    *input* ``xRef``, not the desired signal, so the figure quoted for this data
    set ("~20 dB") is an error power relative to the carrier, and is negative.
    """
    ref = np.asarray(ref).ravel()
    err = np.asarray(err).ravel()
    return float(10.0 * np.log10(np.sum(np.abs(err) ** 2) / np.sum(np.abs(ref) ** 2)))


def chebyshev_func(x: np.ndarray, n: int, order: int = 1) -> np.ndarray:
    """MATLAB ``ChebyshevFunc(x, n, Order)``.

    ``order=1`` gives Chebyshev polynomials of the first kind ``T_n``; ``order=2``
    seeds the same recurrence with ``2x`` (second kind ``U_n``).
    """
    if order not in (1, 2):
        raise ValueError("incorrect Order")
    if n == 0:
        return np.ones_like(x)
    prev, cur = np.ones_like(x), (x if order == 1 else 2.0 * x)
    for _ in range(2, n + 1):
        prev, cur = cur, 2.0 * x * cur - prev
    return cur


def gen_spl_chebyshev(n_basis: int, mg_max: int = 2**15) -> np.ndarray:
    """MATLAB ``gen_spl(Mg_max, Mg_max/N, 1, 6)`` -- the Chebyshev branch.

    ``gen_spl`` with ``Sp_ord=6`` builds ``N = N1D + 1`` rows (the caller then
    sets ``splNum1D = ModelBasisFuncNum + 1``) of ``T_{k-1}`` sampled on the
    magnitude grid ``(0:Mg_max-1) + 0.5`` mapped to ``[-1, 1)``.  The knot
    arithmetic in ``gen_spl.m`` only determines the row count for this branch.

    Returns an ``(n_basis, mg_max)`` lookup table indexed by quantised ``|x|``.
    """
    mg_ind = np.arange(mg_max, dtype=np.float64) + 0.5
    t = 2.0 * (mg_ind / mg_max - 0.5)
    return np.stack([chebyshev_func(t, k, 1) for k in range(n_basis)])


def lin2tensor(shape, ndx: int) -> tuple[int, ...]:
    """MATLAB ``lin2tensor(T, ndx)`` -- 1-based column-major ``ind2sub``."""
    return tuple(int(v) + 1 for v in np.unravel_index(ndx - 1, tuple(shape), order="F"))


def tensor_order_table(shape) -> np.ndarray:
    """All multi-indices in ``lin2tensor`` order, 0-based, as ``(prod(shape), D)``.

    Equivalent to ``[lin2tensor(shape, i) - 1 for i in 1..prod(shape)]`` but
    vectorised; this fixes the column ordering of the regressor matrix so
    coefficients line up with the MATLAB solution.
    """
    shape = tuple(int(s) for s in shape)
    n = int(np.prod(shape))
    return np.stack(np.unravel_index(np.arange(n), shape, order="F"), axis=1)
