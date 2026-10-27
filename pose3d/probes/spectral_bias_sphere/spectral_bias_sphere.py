"""Spectral bias on the sphere: MLP vs CGENN vs GATr fitting sums of spherical harmonics (the experiment of Rahaman et al. 2019, Fig. 1).

Target on S^2:  lambda(x) = sum_{l=1..L} A_l * c_l . Y_l(x), with a random unit vector c_l per degree and real spherical harmonics in
'component' normalisation (mean over the sphere of (c_l . Y_l)^2 = 1), i.e. every degree has power A_l^2: the analogue of
equal-amplitude sinusoids. Models see the point x (a unit vector) and, for the two Clifford models, the constant frame e1, e2, e3, which
breaks the rotation symmetry (an equivariant network that sees only x can only output a constant on the sphere).
Measured: the fraction of each target degree the model has recovered, rho_l(t) = <c_hat_l, A_l c_l> / A_l^2, with c_hat the least-squares
spherical-harmonic coefficients of the model on a dense held-out grid (degrees 0..LMAX, so leakage into other degrees is separated).
"""
import argparse, json, math, sys, time

import numpy as np
import torch
import torch.nn as nn

import os
sys.path.insert(0, os.environ.get("POSE3D_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from pose3d.models.gatr_denoiser import _PGA_INDEX, _ensure_xformers_stub, _import_gatr  # noqa: E402

_ensure_xformers_stub()
from clifford.algebra.cliffordalgebra import CliffordAlgebra  # noqa: E402
from e3nn import o3  # noqa: E402
from pose3d.models.ga_layers import TralaleroTralala  # noqa: E402

torch.set_default_dtype(torch.float32)


def fibonacci_sphere(n, seed=None):
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = torch.sqrt(1 - z * z)
    pts = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)
    if seed is not None:                                   # random global rotation, so train and eval points differ
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
        pts = pts @ q.T
    return pts.float()


def harmonics(x, lmin, lmax):
    return o3.spherical_harmonics(list(range(lmin, lmax + 1)), x.double(), normalize=True, normalization="component").double()


# ---------------------------------------------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, width, depth=6):
        super().__init__()
        dims = [3] + [width] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CGENN(nn.Module):
    """Clifford group equivariant network (the geometric-product MLP used by Clifford Flow) on [x, e1, e2, e3]."""

    def __init__(self, hidden):
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.net = TralaleroTralala(self.algebra, in_features=4, hidden_dim=hidden, out_features=1)
        self.register_buffer("frame", self.algebra.embed(torch.eye(3), (1, 2, 3)))

    def forward(self, x):
        b = x.shape[0]
        mv = torch.cat([self.algebra.embed(x, (1, 2, 3)).unsqueeze(1), self.frame.unsqueeze(0).expand(b, -1, -1)], 1)
        return self.net(mv)[:, 0, 0]                       # scalar grade of the output multivector


class GATrNet(nn.Module):
    """GATr over four tokens [x, e1, e2, e3] (one multivector channel each, one-hot token id as scalars); scalar read-out of token 0."""

    def __init__(self, mv, sc, blocks, heads=4):
        super().__init__()
        GATr = _import_gatr()
        self.net = GATr(in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=mv, in_s_channels=4, out_s_channels=1,
                        hidden_s_channels=sc, attention={"num_heads": heads}, mlp={}, num_blocks=blocks)
        self.register_buffer("_idx", torch.tensor(_PGA_INDEX))
        self.register_buffer("_eye", torch.eye(4))

    def forward(self, x):
        b = x.shape[0]
        mv8 = x.new_zeros(b, 4, 8)
        mv8[:, 0, 1:4] = x
        mv8[:, 1:, 1:4] = torch.eye(3, dtype=x.dtype, device=x.device)
        pga = x.new_zeros(b, 4, 16)
        pga[..., self._idx] = mv8
        _, out_s = self.net(pga.unsqueeze(2), scalars=self._eye.to(x.dtype).expand(b, -1, -1))
        return out_s[:, 0, 0]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(name):
    if name == "mlp":
        return MLP(_match(lambda w: MLP(w), TARGET_PARAMS, range(16, 400)))
    if name == "mlp256":
        return MLP(256)
    if name == "cgenn":
        h = _match(lambda h: CGENN([h]), TARGET_PARAMS, range(4, 200))
        return CGENN([h])
    if name == "gatr":
        return GATrNet(8, 32, 4)
    raise ValueError(name)


TARGET_PARAMS = 97762                                      # GATr, 8 multivector / 32 scalar channels, 4 blocks


def _match(make, target, widths):
    best = min(widths, key=lambda w: abs(n_params(make(w)) - target))
    return best


# ------------------------------------------------------------------------------------------------------------ experiment
def make_target(seed, L, amp):
    g = torch.Generator().manual_seed(1000 + seed)
    blocks = []
    for l in range(1, L + 1):
        c = torch.randn(2 * l + 1, generator=g, dtype=torch.float64)
        blocks.append(c / c.norm())
    a = torch.tensor(amp, dtype=torch.float64)
    return blocks, a


def target_fn(x, blocks, a, L):
    y = harmonics(x, 1, L)
    coef = torch.cat([a[l - 1] * blocks[l - 1] for l in range(1, L + 1)])
    return (y @ coef).float()


def spectrum(model, xe, ye_pinv, L, LMAX, blocks, a):
    """rho_l = <c_hat_l, A_l c_l> / A_l^2 for l = 1..L, and the amplitude |c_hat_l| / A_l."""
    with torch.no_grad():
        f = model(xe).double().cpu()
    coef = ye_pinv @ f                                     # least squares over degrees 0..LMAX
    rho, amp, o = [], [], 1
    for l in range(1, LMAX + 1):
        blk = coef[o:o + 2 * l + 1]
        o += 2 * l + 1
        if l <= L:
            rho.append(float(blk @ (a[l - 1] * blocks[l - 1]) / a[l - 1] ** 2))
            amp.append(float(blk.norm() / a[l - 1]))
    return rho, amp


def run(name, seed, args, amp):
    torch.manual_seed(seed)
    dev = torch.device(args.device)
    model = build(name).to(dev)
    L, LMAX = args.L, args.LMAX
    xt = fibonacci_sphere(args.ntrain, seed=100 + seed)
    xe = fibonacci_sphere(args.neval, seed=200 + seed)
    blocks, a = make_target(seed, L, amp)
    yt, ye = target_fn(xt, blocks, a, L).to(dev), target_fn(xe, blocks, a, L).to(dev)
    xt, xe_cpu = xt.to(dev), xe
    xe = xe.to(dev)
    pinv = torch.linalg.pinv(harmonics(xe_cpu, 0, LMAX))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = sorted(set(int(round(s)) for s in np.geomspace(1, args.steps, args.n_ckpt)) | {0})
    out = {"steps": [], "rho": [], "amp": [], "train_mse": [], "test_mse": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step in steps:
            model.eval()
            rho, amp_l = spectrum(model, xe, pinv, L, LMAX, blocks, a)
            with torch.no_grad():
                out["steps"].append(step); out["rho"].append(rho); out["amp"].append(amp_l)
                out["train_mse"].append(float(((model(xt) - yt) ** 2).mean()))
                out["test_mse"].append(float(((model(xe) - ye) ** 2).mean()))
            model.train()
        if step == args.steps:
            break
        opt.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    out["params"] = n_params(model)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,mlp256,cgenn,gatr")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--LMAX", type=int, default=11)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ntrain", type=int, default=2000)
    ap.add_argument("--neval", type=int, default=3000)
    ap.add_argument("--n_ckpt", type=int, default=36)
    ap.add_argument("--amps", default="equal", choices=["equal", "increasing"])
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    amp = [1.0] * args.L if args.amps == "equal" else list(np.linspace(0.1, 1.0, args.L))
    res = {"args": vars(args), "amp": amp}
    for name in args.models.split(","):
        res[name] = []
        for s in range(args.seed_start, args.seed_start + args.seeds):
            r = run(name, s, args, amp)
            res[name].append(r)
            print(f"{name} seed {s}: params {r['params']}, {r['seconds']:.0f}s, final test mse {r['test_mse'][-1]:.4f}, "
                  f"rho(l=1,5,last) = {r['rho'][-1][0]:.2f} {r['rho'][-1][min(4, len(r['rho'][-1]) - 1)]:.2f} {r['rho'][-1][-1]:.2f}", flush=True)
            json.dump(res, open(args.out, "w"))
