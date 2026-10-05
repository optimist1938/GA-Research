"""Loaders for the two supplied data sets.

``GeoData_TB.mat`` -- tri-band concurrent PA capture, already reduced to the
low-rate model-ready form (``x*``, ``d*``, ``eRef*`` per band).  This is the
data the reference NMSE figure refers to.

``DOV2.mat`` -- the single-band capture used by ``NonLinearProblemSimple.m``;
:func:`preprocess_dov2` reproduces that script's front end (2x upsample,
polyphase split, band-limiting) so the same model code can run on it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import scipy.io as sio

from .matlab import conv_same, fir1, nmse_db

__all__ = ["BandData", "GeoDataTB", "load_geodata_tb", "load_dov2", "preprocess_dov2", "Split"]

# ``BL`` from NonLinearProblemSimple.m: the band-limitation filter applied to
# every nonlinear basis term.  flen = 32 -> 33 taps.
BL_DEFAULT = fir1(32, 0.36 * 2, "low")


@dataclasses.dataclass
class BandData:
    """One carrier of the capture.

    Attributes
    ----------
    x : complex baseband input envelope, shape ``(N,)``.
    d : desired signal -- the band-limited nonlinear residual the model must fit.
    e_ref : residual of the vendor's own reference model, for comparison only.
    carrier_hz : RF carrier frequency.
    """

    name: str
    x: np.ndarray
    d: np.ndarray
    e_ref: np.ndarray
    carrier_hz: float

    @property
    def n(self) -> int:
        return self.x.size

    def nmse_no_model(self) -> float:
        """NMSE if the model output were identically zero."""
        return nmse_db(self.x, self.d)

    def nmse_reference(self) -> float:
        """NMSE of the reference model shipped with the capture."""
        return nmse_db(self.x, self.e_ref)


@dataclasses.dataclass
class GeoDataTB:
    bands: dict[str, BandData]
    fs_hz: float

    def stack(self, order: str) -> np.ndarray:
        """Envelopes of ``order`` (e.g. ``"AB"``) as a ``(len(order), N)`` array."""
        return np.stack([self.bands[b].x for b in order])

    def normalised_stack(self, order: str) -> tuple[np.ndarray, np.ndarray]:
        """As :meth:`stack`, but each row scaled to ``max|x| == 1``.

        ``SimpleNonLinearModel_ML`` indexes the Chebyshev lookup table with
        ``round(|x| * absScale * 2^15)``, so every row must be unit-peak or the
        basis is evaluated off its support.  Returns ``(x_normalised, scales)``;
        the *unscaled* first row is the NMSE reference ``xRef``.
        """
        x = self.stack(order).copy()
        scales = np.array([1.0 / np.abs(row).max() for row in x])
        return x * scales[:, None], scales


def load_geodata_tb(path: str | Path) -> GeoDataTB:
    mat = sio.loadmat(str(path))
    carriers = {"A": float(mat["fa"].ravel()[0]), "B": float(mat["fb"].ravel()[0]), "C": float(mat["fc"].ravel()[0])}
    bands = {
        b: BandData(
            name=b,
            x=mat[f"x{b}"].ravel().astype(np.complex128),
            d=mat[f"d{b}"].ravel().astype(np.complex128),
            e_ref=mat[f"eRef{b}"].ravel().astype(np.complex128),
            carrier_hz=carriers[b],
        )
        for b in "ABC"
    }
    return GeoDataTB(bands=bands, fs_hz=float(mat["FsLow"].ravel()[0]))


def load_dov2(path: str | Path) -> dict[str, np.ndarray]:
    mat = sio.loadmat(str(path))
    return {k: mat[k].ravel() for k in ("PDin", "PDout", "PDdpd")}


def preprocess_dov2(path: str | Path, upsample: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of the front end of ``NonLinearProblemSimple.m``.

    Returns ``(x, d, x_ref, e_ref)`` where ``x`` is the ``(upsample, N)``
    polyphase stack -- note that for DOV2 the "dimensions" of the nonlinearity
    are *fractional delays of one carrier*, not different carriers as in
    GeoData_TB.
    """
    raw = load_dov2(path)
    pd_in, pd_out, pd_dpd = raw["PDin"], raw["PDout"], raw["PDdpd"]
    sml = slice(0, pd_in.size - 1)

    hbf = np.array([0.0, 1.0, 0.0]) if upsample == 1 else fir1(4096, 1.0 / upsample, "low")

    def up_filt(sig: np.ndarray) -> np.ndarray:
        up = np.zeros(sig.size * upsample, dtype=sig.dtype)
        up[::upsample] = sig * upsample
        return np.round(conv_same(up, hbf))

    in_a, err_a, dpd_a = up_filt(pd_in[sml]), up_filt(pd_out[sml]), up_filt(pd_dpd[sml])
    err_a = dpd_a - err_a

    bl = BL_DEFAULT
    e_ref = np.round(conv_same(pd_out[sml] - pd_in[sml], bl))

    x = np.stack([in_a[p::upsample] for p in range(upsample)])
    d = np.round(conv_same(err_a[::upsample], bl))

    g = 2.0**-15
    x, d = x * g, d * g
    x_ref = x[0].copy()
    x = x / np.abs(x).max(axis=1, keepdims=True)
    return x, d, x_ref, e_ref


@dataclasses.dataclass
class Split:
    """Contiguous train/validation split with a guard gap.

    Memory kernels reach ~10 samples and the band-limiting filter another 16, so
    a guard of a few hundred samples removes any leakage between the two halves
    while costing nothing at N = 221000.
    """

    train: slice
    val: slice

    @classmethod
    def contiguous(cls, n: int, train_frac: float = 0.8, guard: int = 512) -> "Split":
        cut = int(n * train_frac)
        return cls(train=slice(0, cut - guard), val=slice(cut + guard, n))
