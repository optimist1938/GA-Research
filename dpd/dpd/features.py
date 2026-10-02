"""Regressors and U(1)^3 invariants built from the three band signals.

Only the three in-band phases matter physically: rotating band b by e^{i a_b} rotates d_b by the
same phase. Everything here is either invariant to those rotations (magnitudes, x_b(k-i) conj
x_b(k-j)) or carries exactly one band's phase (the carrier taps x_c(k-m)).
"""

from __future__ import annotations

import itertools

import numpy as np

from dpd.data import delay


def lut_terms(memory: int) -> list[tuple[int, int]]:
    """(m, l) pairs of x_c(k-m) * LUT(|x(k-l)|): the diagonal m = l plus the l = 0 column."""
    ms = range(-memory, memory + 1)
    return sorted({(m, m) for m in ms} | {(m, 0) for m in ms})


def magnitude_basis(a: np.ndarray, amax: np.ndarray, basis: str, order: int) -> np.ndarray:
    """Per-band 1D basis of magnitudes: a (3, N) -> (3, order + 1, N)."""
    t = np.clip(a / amax[:, None], 0.0, 1.0)
    out = np.empty((a.shape[0], order + 1, a.shape[1]))
    if basis == "chebyshev":
        t = 2 * t - 1
        out[:, 0] = 1.0
        if order >= 1:
            out[:, 1] = t
        for n in range(2, order + 1):
            out[:, n] = 2 * t * out[:, n - 1] - out[:, n - 2]
    elif basis == "even_poly":
        for n in range(order + 1):
            out[:, n] = t ** (2 * n)
    else:
        raise ValueError(basis)
    return out


def tensor_index(basis: str, order: int) -> list[tuple[int, int, int]]:
    """Which (nA, nB, nC) products of the 1D bases make up the 3D LUT."""
    idx = itertools.product(range(order + 1), repeat=3)
    if basis == "even_poly":   # total degree cap, like a polynomial in |xA|^2, |xB|^2, |xC|^2
        return [n for n in idx if sum(n) <= order]
    return list(idx)


def invariants(x: np.ndarray, memory: int, cross_time: bool, max_lag_diff: int) -> np.ndarray:
    """U(1)^3 invariants over taps k-memory..k+memory: (N, F) float32.

    |x_b(k-i)|^2 for every band and tap, and (cross_time) Re / Im of x_b(k-i) conj x_b(k-j),
    1 <= j - i <= max_lag_diff, which carries the phase change of band b between taps.
    """
    taps = {i: delay(x, i) for i in range(-memory, memory + 1)}
    cols = [np.abs(taps[i]) ** 2 for i in taps]                       # each (3, N)
    if cross_time:
        for i in taps:
            for j in range(i + 1, min(i + max_lag_diff, memory) + 1):
                p = taps[i] * np.conj(taps[j])
                cols += [p.real, p.imag]
    return np.concatenate(cols, axis=0).T.astype(np.float32)


def carrier_taps(x: np.ndarray, memory: int) -> np.ndarray:
    """x_c(k-m) for m in -memory..memory: (N, 3, 2 memory + 1) complex64."""
    return np.stack([delay(x, m) for m in range(-memory, memory + 1)], axis=-1).transpose(1, 0, 2).astype(np.complex64)


def poly_features(x: np.ndarray, memory: int) -> np.ndarray:
    """The team's first MLP input: Re / Im of x, x|x|^2, x|x|^4 at taps k..k-memory+1, (N, F).

    Raw Re / Im parts make this NOT equivariant: the network has to learn the phase symmetry.
    """
    cols = []
    for i in range(memory):
        v = delay(x, i)
        p = np.abs(v) ** 2
        for f in (v, v * p, v * p * p):
            cols += [f.real, f.imag]
    return np.concatenate(cols, axis=0).T.astype(np.float32)


def local_invariants(x: np.ndarray, max_lag_diff: int) -> np.ndarray:
    """Per-sample U(1)^3 invariants for a sequence model, (N, 3 (1 + 2 max_lag_diff)) float32:
    |x_b(k)|^2 and Re / Im of x_b(k) conj x_b(k-j), j = 1..max_lag_diff.

    A conv net over this series sees every x_b(k-i) conj x_b(k-j) pair inside its receptive field
    (as sums of the adjacent ones' phases) without the window's quadratic feature count.
    """
    cols = [np.abs(x) ** 2]
    for j in range(1, max_lag_diff + 1):
        p = x * np.conj(delay(x, j))
        cols += [p.real, p.imag]
    return np.concatenate(cols, axis=0).T.astype(np.float32)
