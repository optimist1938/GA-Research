"""Metrics: the NMSE convention, and the spectral diagnostics NMSE cannot give."""

from __future__ import annotations

import numpy as np
import pytest

from dpd.metrics import acpr_db, nmse_db, summarise, welch_psd


def test_nmse_is_negative_when_the_error_is_below_the_reference():
    """Lower is better, and the sign is not a bug -- see vibe/metrics.md."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    assert nmse_db(ref, ref) == pytest.approx(0.0)
    assert nmse_db(ref, ref * 0.1) == pytest.approx(-20.0)
    assert nmse_db(ref, ref * 10) == pytest.approx(20.0)  # error above the carrier


def test_summarise_improvement_is_the_conversion_between_normalisations():
    """improvement_db converts carrier-referred NMSE to desired-referred NMSE.

    ``NMSE_vs_d = NMSE_vs_x - NMSE_no_model``, which is what lets these numbers
    be compared with the literature's ``sum|d-y|^2 / sum|d|^2`` convention.
    """
    rng = np.random.default_rng(1)
    n = 8192
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    d = 0.3 * x + 0.05 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    y = 0.25 * x
    out = summarise(x, d, y)
    direct = nmse_db(d, d - y)
    assert out["nmse_model_db"] - out["nmse_no_model_db"] == pytest.approx(direct)
    assert out["improvement_db"] == pytest.approx(out["nmse_no_model_db"] - out["nmse_model_db"])


def test_welch_psd_locates_a_tone():
    fs, nfft, n = 100.0, 1024, 32768
    f0 = 12.5
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f0 * t)
    f, psd = welch_psd(x, nfft, fs)
    assert f[np.argmax(psd)] == pytest.approx(f0, abs=fs / nfft)
    assert len(f) == nfft


def test_welch_psd_rejects_a_signal_shorter_than_the_window():
    with pytest.raises(ValueError, match="shorter than nfft"):
        welch_psd(np.zeros(64), nfft=1024)


def test_acpr_separates_a_clean_carrier_from_a_spread_one():
    """A narrow carrier must score far lower than one with adjacent-band energy."""
    fs, nfft, n = 100.0, 1024, 65536
    rng = np.random.default_rng(2)
    t = np.arange(n) / fs
    clean = np.exp(2j * np.pi * 1.0 * t)
    spread = clean + 0.5 * np.exp(2j * np.pi * 20.0 * t)
    occupied, offset = 6.0, 20.0
    assert acpr_db(clean, occupied, offset, nfft, fs) < acpr_db(spread, occupied, offset, nfft, fs) - 20


def test_acpr_rejects_an_out_of_range_offset():
    with pytest.raises(ValueError, match="empty in-band or adjacent-band"):
        acpr_db(np.ones(8192), occupied=0.1, offset=10.0, nfft=1024, fs=1.0)
