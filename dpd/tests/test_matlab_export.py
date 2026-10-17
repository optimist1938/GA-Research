"""matlab_model/dpd_gated_tcn.m, mirrored line by line in NumPy, must reproduce the torch model.

The mirror reads the model through a real savemat/loadmat round trip, so the array layouts
MATLAB will see (W1(:,:,j,i), column-vector biases, the 42-output reshape) are what is tested.
"""

import pathlib

import numpy as np
import pytest
import scipy.io as sio
import torch
from torch import nn

from dpd.export_matlab import matlab_struct, predict
from dpd.models import GatedTCN

MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "matlab_model"


def _checkpoint(levels=2, channels=8, memory=2, lag=2, seed=0):
    torch.manual_seed(seed)
    n_inv = 3 * (1 + 2 * lag)
    net = GatedTCN(n_inv, 2 * memory + 1, channels, levels)
    for p in net.parameters():          # the output layer starts at zero: randomize everything
        nn.init.normal_(p, std=0.2)
    rng = np.random.default_rng(seed)
    ck = {"kind": "gated_tcn",
          "config": {"net": {"max_lag_diff": lag, "tcn_channels": channels, "tcn_levels": levels},
                     "model": {"memory": memory}},
          "state_dict": net.state_dict(),
          "mu": torch.tensor(rng.normal(size=n_inv), dtype=torch.float32),
          "sd": torch.tensor(rng.uniform(0.5, 2.0, size=n_inv), dtype=torch.float32),
          "scale": rng.uniform(0.5, 3.0, size=3),
          "n_params": sum(p.numel() for p in net.parameters())}
    return ck, net.eval()


def _signal(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))) * 0.4


def _as_matlab(model, tmp_path):
    f = tmp_path / "model.mat"
    sio.savemat(f, {"model": model})
    return sio.loadmat(f, struct_as_record=False, squeeze_me=False)["model"][0, 0]


# ---------------------------------------------------------------- NumPy mirror of the .m file

def _s(a):
    return float(np.asarray(a).ravel()[0])


def zdelay(x, k):
    n = x.shape[1]
    y = np.zeros(x.shape, dtype=x.dtype)
    if abs(k) >= n:
        return y
    if k >= 0:
        y[:, k:n] = x[:, 0:n - k]
    else:
        y[:, 0:n + k] = x[:, -k:n]
    return y


def gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def conv_dil3(a, W, b, dil):
    y = np.repeat(b, a.shape[1], axis=1)
    for j in range(1, 4):
        y = y + W[:, :, j - 1] @ zdelay(a, -(j - 2) * dil)
    return y


def local_invariants(X, J):
    F = [np.abs(X) ** 2]
    for j in range(1, J + 1):
        p = X * np.conj(zdelay(X, j))
        F += [p.real, p.imag]
    return np.concatenate(F, axis=0)


def m_forward(X, m):
    n = X.shape[1]
    memory, levels = int(_s(m.memory)), int(_s(m.levels))
    X = X * m.scale
    F = (local_invariants(X, int(_s(m.max_lag_diff))) - m.mu) / m.sd
    h = m.W_in @ F + m.b_in
    W1 = m.W1.reshape(m.W1.shape[:3] + (levels,))
    for i in range(1, levels + 1):
        a = gelu(h)
        a = conv_dil3(a, W1[:, :, :, i - 1], m.b1[:, i - 1:i], 2 ** (i - 1))
        a = gelu(a)
        a = m.W2.reshape(m.W2.shape[:2] + (levels,))[:, :, i - 1] @ a + m.b2[:, i - 1:i]
        h = h + a
    g = m.W_out @ h + m.b_out
    T = 2 * memory + 1
    G = g.reshape((2, T, 3, n), order="F")
    gain = G[0] + 1j * G[1]                     # (T, 3, n)
    D = np.zeros((3, n), dtype=complex)
    for t in range(1, T + 1):
        D = D + zdelay(X, t - 1 - memory) * gain[t - 1]
    return D / m.scale


def m_dpd(X, m, chunk=65536):
    N = X.shape[1]
    halo = int(_s(m.receptive) + _s(m.memory) + _s(m.max_lag_diff) + 2)
    D = np.zeros((3, N), dtype=complex)
    for s in range(1, N + 1, chunk):           # 1-based like the .m file
        e = min(s + chunk - 1, N)
        lo, hi = max(1, s - halo), min(N, e + halo)
        Dc = m_forward(X[:, lo - 1:hi], m)
        D[:, s - 1:e] = Dc[:, s - lo:e - lo + 1]
    return D


# ---------------------------------------------------------------- tests

@pytest.mark.parametrize("levels, channels, memory, lag", [(2, 8, 2, 2), (3, 6, 3, 1), (4, 16, 3, 2)])
def test_mirror_matches_torch(tmp_path, levels, channels, memory, lag):
    ck, net = _checkpoint(levels, channels, memory, lag)
    m = _as_matlab(matlab_struct(ck, net), tmp_path)
    x = _signal()
    want = predict(ck, net, x)
    got = m_dpd(x, m)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5


def test_chunked_equals_whole(tmp_path):
    ck, net = _checkpoint(levels=3)
    m = _as_matlab(matlab_struct(ck, net), tmp_path)
    x = _signal(n=2000)
    whole = m_dpd(x, m)
    for chunk in (97, 500, 1999):
        np.testing.assert_allclose(m_dpd(x, m, chunk), whole, rtol=0, atol=1e-12 * np.abs(whole).max())


def test_mirror_is_phase_equivariant(tmp_path):
    ck, net = _checkpoint()
    m = _as_matlab(matlab_struct(ck, net), tmp_path)
    x = _signal()
    phase = np.exp(1j * np.array([0.7, -1.9, 2.8]))[:, None]
    np.testing.assert_allclose(m_dpd(x * phase, m), m_dpd(x, m) * phase, atol=1e-10)


@pytest.mark.parametrize("name", ["dpd_tcn16x4.mat", "dpd_tcn64x5.mat"])
def test_exported_model_reproduces_its_check_vector(name):
    path = MODEL_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not exported (weights are git-ignored)")
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=False)["model"][0, 0]
    got = m_dpd(m.x_check, m)
    assert np.abs(got - m.d_check).max() / np.abs(m.d_check).max() < 1e-4
