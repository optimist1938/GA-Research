"""The four models of the first CPU round. Each one works on normalized signals x, d of shape (3, N):

    model.fit(x, d, train, test)   # train / test are slices of the series
    model.predict(x) -> (3, N) complex
    model.n_params                 # real parameters (a complex coefficient counts twice)
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from dpd.config import Config
from dpd.data import delay
from dpd.features import (carrier_taps, invariants, local_invariants, lut_terms, magnitude_basis,
                          poly_features, tensor_index)
from dpd.metrics import nmse_db


class LutLS:
    """Huawei's reference: d_c(k) = sum_(m,l) x_c(k-m) * sum_n c_n B_nA(|xA(k-l)|) B_nB(|xB(k-l)|) B_nC(|xC(k-l)|).

    Linear in the coefficients, solved by (ridge) least squares; the Gram matrix is accumulated
    in row chunks so the regressor matrix never has to fit in memory.
    """

    CHUNK = 8192

    def __init__(self, cfg: Config):
        self.terms = lut_terms(cfg.model.memory)
        self.basis, self.order, self.ridge = cfg.lut.lut_basis, cfg.lut.lut_order, cfg.lut.ridge
        self.idx = np.array(tensor_index(self.basis, self.order)).T      # (3, n_idx)
        self.coef = None

    @property
    def n_params(self) -> int:
        return 2 * 3 * len(self.terms) * self.idx.shape[1]

    def _prepare(self, x):
        ls = sorted({l for _, l in self.terms})
        self._basis = {l: magnitude_basis(np.abs(delay(x, l)), self.amax, self.basis, self.order) for l in ls}
        self._carrier = {m: delay(x, m) for m, _ in self.terms}

    def _rows(self, c: int, rows: slice) -> np.ndarray:
        blocks = []
        for m, l in self.terms:
            b = self._basis[l][:, :, rows]
            lut = b[0][self.idx[0]] * b[1][self.idx[1]] * b[2][self.idx[2]]    # (n_idx, len)
            blocks.append(lut.T * self._carrier[m][c, rows][:, None])
        return np.concatenate(blocks, axis=1)

    def _chunks(self, span: slice):
        for s in range(span.start, span.stop, self.CHUNK):
            yield slice(s, min(s + self.CHUNK, span.stop))

    def fit(self, x, d, train, test):
        self.amax = np.abs(x[:, train]).max(axis=1)
        self._prepare(x)
        self.coef = []
        for c in range(3):
            t0 = time.time()
            p = self.n_params // 6
            gram, rhs = np.zeros((p, p), complex), np.zeros(p, complex)
            for rows in self._chunks(train):
                u = self._rows(c, rows)
                gram += u.conj().T @ u
                rhs += u.conj().T @ d[c, rows]
            gram[np.diag_indices(p)] += self.ridge * np.real(np.trace(gram)) / p
            self.coef.append(np.linalg.solve(gram, rhs))
            print(f"  band {c}: {p} complex coefficients, {time.time() - t0:.1f}s")

    def predict(self, x):
        self._prepare(x)
        y = np.zeros_like(x)
        for c in range(3):
            for rows in self._chunks(slice(0, x.shape[1])):
                y[c, rows] = self._rows(c, rows) @ self.coef[c]
        return y


# ----------------------------------------------------------------------------- torch models


def _mlp(n_in: int, hidden: int, layers: int, n_out: int) -> nn.Sequential:
    mods, width = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    last = nn.Linear(width, n_out)
    nn.init.zeros_(last.weight)   # start from the zero model: d is the small nonlinear part
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class GatedNet(nn.Module):
    """U(1)^3-equivariant: d_c(k) = sum_m x_c(k-m) g_{c,m}(invariants(k)), g complex-valued.

    The phase of band c enters only through the carrier taps x_c(k-m); g sees invariants only,
    so rotating each band by its own phase rotates each output by the same phase, exactly.
    """

    def __init__(self, n_inv: int, n_taps: int, hidden: int, layers: int, per_band: bool = False):
        super().__init__()
        self.n_taps = n_taps
        if per_band:
            self.g = nn.ModuleList([_mlp(n_inv, hidden, layers, n_taps * 2) for _ in range(3)])
        else:
            self.g = _mlp(n_inv, hidden, layers, 3 * n_taps * 2)

    def forward(self, inv, taps):
        g = torch.stack([f(inv) for f in self.g], 1) if isinstance(self.g, nn.ModuleList) else self.g(inv)
        g = g.view(-1, 3, self.n_taps, 2)
        return (taps * torch.complex(g[..., 0], g[..., 1])).sum(-1)


class GatedTCN(nn.Module):
    """GatedNet with g a dilated 1D conv net over the per-sample invariant series.

    inv (B, L, F), taps (B, L, 3, T) -> (B, L, 3). Weights are shared over time, so the receptive
    field (+-(2^levels - 1) samples) grows without growing the parameter count.
    """

    def __init__(self, n_inv: int, n_taps: int, channels: int, levels: int):
        super().__init__()
        self.n_taps = n_taps
        self.inp = nn.Conv1d(n_inv, channels, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.GELU(), nn.Conv1d(channels, channels, 3, dilation=2 ** i, padding=2 ** i),
                          nn.GELU(), nn.Conv1d(channels, channels, 1))
            for i in range(levels)])
        self.out = nn.Conv1d(channels, 3 * n_taps * 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.receptive = 2 ** levels - 1

    def forward(self, inv, taps):
        h = self.inp(inv.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        g = self.out(h).transpose(1, 2).reshape(*taps.shape[:2], 3, self.n_taps, 2)
        return (taps * torch.complex(g[..., 0], g[..., 1])).sum(-1)


class MLPPoly(nn.Module):
    """The team's first model: polynomial features of x and conj(x) -> MLP -> Re / Im of d."""

    def __init__(self, n_in: int, hidden: int, layers: int):
        super().__init__()
        self.f = _mlp(n_in, hidden, layers, 6)

    def forward(self, feats):
        y = self.f(feats).view(-1, 3, 2)
        return torch.complex(y[..., 0], y[..., 1])


class LutCP(nn.Module):
    """The 3D LUT of LutLS as a rank-R CP product of 1D functions, per band c and term (m, l):

        LUT(a, b, c) ~= sum_r f_rA(a) f_rB(b) f_rC(c),   f = sum_n w_n B_n

    3 R (K) complex weights per term instead of K^3 (K = lut_order + 1).
    """

    def __init__(self, terms, ls, rank: int, k: int):
        super().__init__()
        self.register_buffer("l_of_term", torch.tensor([ls.index(l) for _, l in terms]))
        t = len(terms)
        w = 0.1 * torch.randn(3, t, rank, 3, k, dtype=torch.cfloat)
        w[..., 0] += 1.0
        w[:, :, :, 0] *= 1e-2 / (t * rank)
        self.w = nn.Parameter(w)

    def forward(self, basis, taps):
        # basis (B, L, 3, K) real, taps (B, 3, T) complex: carrier x_c(k - m_t) of every term t
        b = basis[:, self.l_of_term].to(torch.cfloat)                 # (B, T, 3, K)
        f = torch.einsum("ntbk,ctrbk->nctrb", b, self.w)              # (B, 3, T, R, 3)
        lut = f.prod(-1).sum(-1)                                      # (B, 3, T)
        return (taps * lut).sum(-1)


class TorchModel:
    """Wraps a torch module with its input tensors, the training loop and the parameter count."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.net = None

    @property
    def n_params(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.net.parameters())

    # -- inputs per model
    def _inputs(self, x: np.ndarray) -> dict:
        cfg, kind = self.cfg, self.cfg.model.model
        if kind == "gated":
            inv = invariants(x, cfg.net.inv_memory, cfg.net.cross_time_invariants, cfg.net.max_lag_diff)
            return {"inv": inv, "taps": carrier_taps(x, cfg.model.memory)}
        if kind == "gated_tcn":
            return {"inv": local_invariants(x, cfg.net.max_lag_diff), "taps": carrier_taps(x, cfg.model.memory)}
        if kind == "mlp_poly":
            return {"feats": poly_features(x, cfg.net.poly_memory)}
        if kind == "lut_cp":
            terms = lut_terms(cfg.model.memory)
            ls = sorted({l for _, l in terms})
            basis = np.stack([magnitude_basis(np.abs(delay(x, l)), self.amax, cfg.lut.lut_basis,
                                              cfg.lut.lut_order) for l in ls])   # (L, 3, K, N)
            taps = np.stack([delay(x, m) for m, _ in terms], axis=-1).transpose(1, 0, 2)
            return {"basis": basis.transpose(3, 0, 1, 2).astype(np.float32),
                    "taps": taps.astype(np.complex64)}
        raise ValueError(kind)

    def _build(self, inputs: dict):
        cfg, kind = self.cfg, self.cfg.model.model
        if kind == "gated":
            return GatedNet(inputs["inv"].shape[1], 2 * cfg.model.memory + 1, cfg.net.hidden, cfg.net.layers,
                            cfg.net.per_band)
        if kind == "gated_tcn":
            return GatedTCN(inputs["inv"].shape[1], 2 * cfg.model.memory + 1, cfg.net.tcn_channels,
                            cfg.net.tcn_levels)
        if kind == "mlp_poly":
            return MLPPoly(inputs["feats"].shape[1], cfg.net.hidden, cfg.net.layers)
        terms = lut_terms(cfg.model.memory)
        return LutCP(terms, sorted({l for _, l in terms}), cfg.lut.cp_rank, cfg.lut.lut_order + 1)

    def _tensors(self, x):
        inputs = {k: torch.from_numpy(v) for k, v in self._inputs(x).items()}
        for k in ("inv", "feats"):   # standardize real network inputs with train statistics
            if k in inputs:
                inputs[k] = (inputs[k] - self.mu) / self.sd
        return inputs

    def fit(self, x, d, train, test):
        cfg, tc = self.cfg, self.cfg.train
        self.amax = np.abs(x[:, train]).max(axis=1)
        raw = self._inputs(x)
        for k in ("inv", "feats"):
            if k in raw:
                self.mu = torch.from_numpy(raw[k][train].mean(0))
                self.sd = torch.from_numpy(raw[k][train].std(0) + 1e-6)
        del raw
        inputs = self._tensors(x)
        self.net = self._build(inputs)
        target = torch.from_numpy(d.T.astype(np.complex64))             # (N, 3)
        power = target[train].abs().pow(2).mean(0)                      # per-band normalization
        weight = torch.ones_like(target.real)
        if cfg.model.model == "mlp_poly":   # the team's |d|-weighted MSE, to fit the peaks
            mag = target.abs()
            weight = 1 + mag / mag[train].mean(0)

        opt = torch.optim.Adam(self.net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        batches = self._sequence_batches if self.sequential else self._sample_batches
        steps = tc.epochs * sum(1 for _ in batches(train))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=steps, pct_start=0.05)
        for epoch in range(tc.epochs):
            t0, total, count = time.time(), 0.0, 0
            self.net.train()
            for b, keep in batches(train):
                pred = self.net(**{k: v[b] for k, v in inputs.items()})[keep]
                bk = b[keep]
                loss = ((pred - target[bk]).abs().pow(2) * weight[bk] / power).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total, count = total + loss.item() * bk.shape[0], count + bk.shape[0]
            if epoch % 5 == 4 or epoch == tc.epochs - 1:
                y = self._predict_inputs(inputs)
                test_db = [nmse_db(d[c, test] - y[c, test], d[c, test]) for c in range(3)]
                print(f"  epoch {epoch + 1:3d}  train loss {total / count:.4f}  "
                      f"test NMSE vs d {' '.join(f'{v:6.2f}' for v in test_db)} dB  {time.time() - t0:.1f}s/epoch")

    @property
    def sequential(self) -> bool:
        return self.cfg.model.model == "gated_tcn"

    def _sample_batches(self, train):
        """Shuffled single samples; keep = everything."""
        idx = torch.arange(train.start, train.stop)
        for b in idx[torch.randperm(len(idx))].split(self.cfg.train.batch_size):
            yield b, slice(None)

    def _sequence_batches(self, train):
        """Random windows (B, L) inside the train span; the loss skips each window's edges, where
        the conv net's receptive field would run past the window."""
        length, margin = self.cfg.net.tcn_window, self.net.receptive + self.cfg.model.memory
        n_windows = max(1, self.cfg.train.batch_size // 128)
        for _ in range((train.stop - train.start) // length):
            starts = torch.randint(train.start, train.stop - length + 1, (n_windows,))
            yield starts[:, None] + torch.arange(length), (slice(None), slice(margin, length - margin))

    @torch.no_grad()
    def _predict_inputs(self, inputs) -> np.ndarray:
        self.net.eval()
        if self.sequential:   # the whole series is one sequence
            return self.net(**{k: v[None] for k, v in inputs.items()})[0].numpy().T.astype(complex)
        n = next(iter(inputs.values())).shape[0]
        out = [self.net(**{k: v[s:s + 8192] for k, v in inputs.items()}) for s in range(0, n, 8192)]
        return torch.cat(out).numpy().T.astype(complex)

    def predict(self, x):
        return self._predict_inputs(self._tensors(x))


class ResidualOnLut:
    """lut_ls first, then `top` on its residual d - lut(x); the output is their sum."""

    def __init__(self, cfg: Config, top):
        self.base, self.top = LutLS(cfg), top

    @property
    def n_params(self) -> int:
        return self.base.n_params + self.top.n_params

    def fit(self, x, d, train, test):
        self.base.fit(x, d, train, test)
        y0 = self.base.predict(x)
        print(f"  lut_ls test NMSE vs d {' '.join(f'{nmse_db(d[c, test] - y0[c, test], d[c, test]):6.2f}' for c in range(3))} dB"
              " (the rows below are vs the residual)")
        self.top.fit(x, d - y0, train, test)

    def predict(self, x):
        return self.base.predict(x) + self.top.predict(x)


def build(cfg: Config):
    if cfg.model.model == "lut_ls":
        return LutLS(cfg)
    model = TorchModel(cfg)
    return ResidualOnLut(cfg, model) if cfg.model.residual_lut else model
