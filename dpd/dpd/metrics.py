"""Evaluation metrics: NMSE and spectral (PSD / ACPR) diagnostics."""

from __future__ import annotations

import numpy as np

from .matlab import nmse_db

__all__ = ["nmse_db", "welch_psd", "acpr_db", "summarise"]


def welch_psd(x: np.ndarray, nfft: int = 2048, fs: float = 1.0):
    """Welch PSD with 50% overlap and a Hann window; returns ``(f, psd_db)``.

    Mirrors what ``plot_psd4.m`` shows, which is the qualitative check that
    matters for a PA model: NMSE can look fine while the out-of-band shoulders
    are untouched.
    """
    x = np.asarray(x).ravel()
    if len(x) < nfft:
        raise ValueError(f"signal of length {len(x)} is shorter than nfft={nfft}")
    win = np.hanning(nfft)
    step = nfft // 2
    segs = [x[i : i + nfft] * win for i in range(0, len(x) - nfft + 1, step)]
    psd = np.mean([np.abs(np.fft.fftshift(np.fft.fft(s))) ** 2 for s in segs], axis=0)
    psd /= np.sum(win**2) * fs
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return f, 10 * np.log10(psd + np.finfo(float).tiny)


def acpr_db(x: np.ndarray, occupied: float, offset: float, nfft: int = 2048, fs: float = 1.0) -> float:
    """Adjacent-channel power ratio: adjacent-band power over in-band power, dB.

    ``occupied`` and ``offset`` are in the same units as ``fs``.
    """
    f, psd_db = welch_psd(x, nfft, fs)
    psd = 10 ** (psd_db / 10)
    in_band = np.abs(f) <= occupied / 2
    adj = (np.abs(f) >= offset - occupied / 2) & (np.abs(f) <= offset + occupied / 2)
    if not adj.any() or not in_band.any():
        raise ValueError("empty in-band or adjacent-band selection; check occupied/offset vs fs")
    return float(10 * np.log10(psd[adj].sum() / psd[in_band].sum()))


def summarise(x_ref: np.ndarray, d: np.ndarray, y: np.ndarray, e_ref: np.ndarray | None = None) -> dict:
    """Standard result row: model NMSE, the no-model floor, and the reference."""
    out = {
        "nmse_model_db": nmse_db(x_ref, d - y),
        "nmse_no_model_db": nmse_db(x_ref, d),
    }
    if e_ref is not None:
        out["nmse_reference_db"] = nmse_db(x_ref, e_ref)
    out["improvement_db"] = out["nmse_no_model_db"] - out["nmse_model_db"]
    return out
