from __future__ import annotations

import numpy as np
from scipy.signal import welch


def nmse_db(err: np.ndarray, ref: np.ndarray) -> float:
    """10 log10(sum|err|^2 / sum|ref|^2).

    Huawei's simpleModel normalizes by the input x (`nmse(xRef, e)`); we report both that and
    the normalization by d, which is ~11-16 dB stricter here because d is much weaker than x.
    """
    return float(10 * np.log10(np.sum(np.abs(err) ** 2) / np.sum(np.abs(ref) ** 2)))


def band_project(y: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero every FFT bin of y (..., N) with |f| > cutoff_hz."""
    spec = np.fft.fft(y, axis=-1)
    spec[..., np.abs(np.fft.fftfreq(y.shape[-1], 1 / fs)) > cutoff_hz] = 0
    return np.fft.ifft(spec, axis=-1)


def psd_plot(path, fs: float, curves: dict[str, np.ndarray], title: str, nperseg: int = 2048) -> None:
    """Two-sided Welch PSD of each curve, as in Huawei's plot_psd4 (x, d, model error, ...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, v in curves.items():
        f, p = welch(v, fs=fs, nperseg=nperseg, return_onesided=False, detrend=False)
        order = np.argsort(f)
        ax.plot(f[order] / 1e6, 10 * np.log10(p[order] + 1e-30), label=label, lw=0.8)
    ax.set_xlabel("frequency, MHz (baseband)")
    ax.set_ylabel("PSD, dB/Hz")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
