"""The Huawei 3-band data set (GeoData_TB.mat).

x[b] is the complex baseband input of band b (A, B, C), d[b] the desired DPD correction
(the nonlinear part only: it has no linear component in x) and e_ref[b] the PA output error
after Huawei's own model, shown on plots only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.io as sio

BANDS = ("A", "B", "C")


@dataclass
class Signals:
    x: np.ndarray        # (3, N) complex
    d: np.ndarray        # (3, N) complex
    e_ref: np.ndarray    # (3, N) complex
    fs: float            # sample rate, Hz
    carriers: tuple      # (fa, fb, fc), Hz

    @property
    def n(self) -> int:
        return self.x.shape[1]


def load(path: str) -> Signals:
    m = sio.loadmat(path)
    stack = lambda prefix: np.stack([m[prefix + b].ravel() for b in BANDS])
    return Signals(
        x=stack("x"), d=stack("d"), e_ref=stack("eRef"),
        fs=float(m["FsLow"].item()),
        carriers=tuple(float(m[k].item()) for k in ("fa", "fb", "fc")),
    )


def split_index(n: int, train_frac: float) -> tuple[slice, slice]:
    cut = int(n * train_frac)
    return slice(0, cut), slice(cut, n)


def delay(v: np.ndarray, k: int) -> np.ndarray:
    """out[..., n] = v[..., n - k], zero-padded (k < 0 looks ahead)."""
    out = np.zeros_like(v)
    if k == 0:
        out[...] = v
    elif k > 0:
        out[..., k:] = v[..., :-k]
    else:
        out[..., :k] = v[..., -k:]
    return out
