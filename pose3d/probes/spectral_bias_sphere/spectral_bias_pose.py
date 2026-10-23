"""Pose-dependent spectral bias: MLP vs CGENN vs GATr fitting lambda(x, R) = sum_l A_l c_l . Y_l(R^T x)  (x a point of S^2, R a rotation).

The target is invariant under turning point and pose together, so equivariant networks need no symmetry-breaking frame: they see x and the
pose as the frame (R e1) e123, R e2, R e3, exactly like P1's pose token, and <x, R e_k> = (R^T x)_k hands them the coordinates of u = R^T x.
The MLP sees the 12 raw numbers (x, R). The spectrum is measured on u: model outputs on held-out random (x, R) are fitted with spherical
harmonics in u = R^T x (degrees 0..LMAX); rho_l is the recovered fraction of the target's degree l (as in spectral_bias_sphere.py).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectral_bias_sphere as S  # noqa: E402  (shares the helpers and the POSE3D_PATH handling)
from e3nn import o3  # noqa: E402
from clifford.algebra.cliffordalgebra import CliffordAlgebra  # noqa: E402
from pose3d.models.ga_layers import TralaleroTralala  # noqa: E402
from pose3d.models.gatr_denoiser import _PGA_INDEX, _import_gatr  # noqa: E402


def frame_mv(R, algebra):
    """(B, 3, 8) Cl(3,0) multivectors: (R e1) e123 as a bivector, then R e2 and R e3 as vectors (columns of R)."""
    a, b, c = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    out = R.new_zeros(R.shape[0], 3, 8)
    out[:, 0, 4], out[:, 0, 5], out[:, 0, 6] = a[:, 2], -a[:, 1], a[:, 0]      # e12, e13, e23
    out[:, 1, 1:4], out[:, 2, 1:4] = b, c
    return out


class MLP(nn.Module):
    def __init__(self, width, depth=6):
        super().__init__()
        dims = [12] + [width] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x, R):
        return self.net(torch.cat([x, R.flatten(1)], 1)).squeeze(-1)


class CGENN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.net = TralaleroTralala(self.algebra, in_features=4, hidden_dim=hidden, out_features=1)

    def forward(self, x, R):
        mv = torch.cat([self.algebra.embed(x, (1, 2, 3)).unsqueeze(1), frame_mv(R, self.algebra)], 1)   # (B, 4, 8)
        return self.net(mv)[:, 0, 0]


class GATrNet(nn.Module):
    """Two tokens: the pose (three multivector channels) and x (channel 0); scalar read-out of the pose token, as in P1."""

    def __init__(self, mv, sc, blocks, heads=4):
        super().__init__()
        GATr = _import_gatr()
        self.net = GATr(in_mv_channels=3, out_mv_channels=1, hidden_mv_channels=mv, in_s_channels=2, out_s_channels=1,
                        hidden_s_channels=sc, attention={"num_heads": heads}, mlp={}, num_blocks=blocks)
        self.register_buffer("_idx", torch.tensor(_PGA_INDEX))
        self.register_buffer("_eye", torch.eye(2))
        self.algebra = CliffordAlgebra((1, 1, 1))

    def forward(self, x, R):
        b = x.shape[0]
        mv8 = x.new_zeros(b, 2, 3, 8)
        mv8[:, 0] = frame_mv(R, self.algebra)
        mv8[:, 1, 0, 1:4] = x
        pga = x.new_zeros(b, 2, 3, 16)
        pga[..., self._idx] = mv8
        _, out_s = self.net(pga, scalars=self._eye.to(x.dtype).expand(b, -1, -1))
        return out_s[:, 0, 0]


TARGET_PARAMS = 97762


def build(name):
    n = S.n_params
    if name == "mlp":
        return MLP(S._match(lambda w: MLP(w), TARGET_PARAMS, range(16, 400)))
    if name == "mlp256":
        return MLP(256)
    if name == "cgenn":
        return CGENN([S._match(lambda h: CGENN([h]), TARGET_PARAMS, range(4, 200))])
    if name == "gatr":
        return GATrNet(8, 32, 4)
    raise ValueError(name)


def sample(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)
    torch.manual_seed(seed)
    R = o3.rand_matrix(n).double()
    return x.float(), R.float(), (R.transpose(1, 2) @ x.unsqueeze(-1)).squeeze(-1).float()      # u = R^T x


def run(name, seed, args, amp):
    torch.manual_seed(seed)
    dev = torch.device(args.device)
    model = build(name).to(dev)
    L, LMAX = args.L, args.LMAX
    xt, Rt, ut = sample(args.ntrain, 100 + seed)
    xe, Re, ue = sample(args.neval, 200 + seed)
    blocks, a = S.make_target(seed, L, amp)
    yt, ye = S.target_fn(ut, blocks, a, L).to(dev), S.target_fn(ue, blocks, a, L).to(dev)
    xt, Rt, xe_d, Re_d = xt.to(dev), Rt.to(dev), xe.to(dev), Re.to(dev)
    pinv = torch.linalg.pinv(S.harmonics(ue, 0, LMAX))                    # spectrum is taken over u = R^T x
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = sorted(set(int(round(s)) for s in np.geomspace(1, args.steps, args.n_ckpt)) | {0})
    out = {"steps": [], "rho": [], "amp": [], "train_mse": [], "test_mse": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step in steps:
            model.eval()
            with torch.no_grad():
                f = model(xe_d, Re_d).double().cpu()
                coef = pinv @ f
                rho, amp_l, o = [], [], 1
                for l in range(1, LMAX + 1):
                    blk = coef[o:o + 2 * l + 1]
                    o += 2 * l + 1
                    if l <= L:
                        rho.append(float(blk @ (a[l - 1] * blocks[l - 1]) / a[l - 1] ** 2))
                        amp_l.append(float(blk.norm() / a[l - 1]))
                out["steps"].append(step); out["rho"].append(rho); out["amp"].append(amp_l)
                out["train_mse"].append(float(((model(xt, Rt) - yt) ** 2).mean()))
                out["test_mse"].append(float(((model(xe_d, Re_d) - ye) ** 2).mean()))
            model.train()
        if step == args.steps:
            break
        opt.zero_grad()
        loss = ((model(xt, Rt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    out["params"] = S.n_params(model)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,mlp256,cgenn,gatr")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--LMAX", type=int, default=11)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ntrain", type=int, default=4000)
    ap.add_argument("--neval", type=int, default=3000)
    ap.add_argument("--n_ckpt", type=int, default=40)
    ap.add_argument("--amps", default="equal", choices=["equal", "increasing"])
    ap.add_argument("--out", default="results_pose.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    amp = [1.0] * args.L if args.amps == "equal" else list(np.linspace(0.1, 1.0, args.L))
    res = {"args": vars(args), "amp": amp}
    for name in args.models.split(","):
        res[name] = []
        for s in range(args.seeds):
            r = run(name, s, args, amp)
            res[name].append(r)
            print(f"{name} seed {s}: params {r['params']}, {r['seconds']:.0f}s, final test mse {r['test_mse'][-1]:.4f}, "
                  f"rho(l=1,5,last) = {r['rho'][-1][0]:.2f} {r['rho'][-1][min(4, len(r['rho'][-1]) - 1)]:.2f} {r['rho'][-1][-1]:.2f}", flush=True)
            json.dump(res, open(args.out, "w"))
