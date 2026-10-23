"""Batched (ensemble) version of the sphere spectral-bias experiment: all (lr, seed) runs of a model are trained at once.

Same idea and the same CGENN layer re-implementation as optimist1938's probes/spectral_bias/ensemble.py (every weight gets a leading
ensemble dim E, the loss is the sum of the per-run MSEs over disjoint weights, Adam is elementwise with a per-run lr, the whole step is
captured in one CUDA graph); here the inputs are points of S^2 (plus the constant frame for CGENN), the target is
lambda(x) = sum_{l=1..L} c_l . Y_l(x), and the spectrum is the recovered fraction rho_l of every target degree on a held-out grid.
All members share the training and evaluation points (the targets, i.e. the c_l, and the initialisations differ per seed).

  python spectral_ens.py --model cgenn --lrs 1e-4 3e-4 1e-3 --seeds 10 --steps 30000 --out results.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectral_bias_sphere as S  # noqa: E402


# ---------------------------------------------------------------- CGENN layers with a leading ensemble dim (after probes/spectral_bias/ensemble.py)
class _GA(nn.Module):
    """The algebra's constants: B[p, i, j, k] = cayley[i, j, k] on product path p, the quadratic form Q of q(x), blade -> grade."""

    def __init__(self, algebra):
        super().__init__()
        cay = algebra.cayley.detach().clone()
        D = cay.shape[0]
        bg = algebra.bbo_grades.long().cpu()
        G = len(algebra.grades)
        paths = algebra.geometric_product_paths.cpu()
        P = int(paths.sum())
        pidx = torch.full((G, G, G), -1, dtype=torch.long)
        pidx[paths] = torch.arange(P)
        B = torch.zeros(P, D, D, D, dtype=cay.dtype)
        for i in range(D):
            for j in range(D):
                for k in range(D):
                    p = int(pidx[bg[i], bg[j], bg[k]])
                    if p >= 0:
                        B[p, i, j, k] = cay[i, j, k]
        beta = algebra._beta_signs.to(cay.dtype).cpu()
        same = (bg[:, None] == bg[None, :]).to(cay.dtype)
        self.register_buffer("B", B)
        self.register_buffer("Q", beta[:, None] * cay[:, 0, :].cpu() * same)
        self.register_buffer("M", F.one_hot(bg, G).to(cay.dtype))
        self.register_buffer("MT", self.M.T.contiguous())
        self.register_buffer("Bm", B.permute(1, 3, 0, 2).reshape(D * D, P * D).contiguous())
        self.register_buffer("bg", bg)
        self.register_buffer("e0", F.one_hot(torch.tensor(0), D).to(cay.dtype))
        self.register_buffer("qd", torch.diagonal(self.Q).contiguous())
        self.D, self.G, self.P = D, G, P
        self.trivial = G == D and bool((bg == torch.arange(D)).all())

    def q(self, x):
        if self.trivial:
            return x * x * self.qd
        return (x * (x @ self.Q.T)) @ self.M

    def expand(self, t):
        if self.trivial:
            return t
        return t @ self.MT

    def pairs(self, x, y):
        D = self.D
        xy = (x[..., :, None] * y[..., None, :]).reshape(*x.shape[:-1], D * D)
        return (xy @ self.Bm).reshape(*x.shape[:-1], self.P, D)


class _Layer(nn.Module):
    def __init__(self, ga):
        super().__init__()
        self._ga = [ga]

    @property
    def ga(self):
        return self._ga[0]


class EnsMVLinear(_Layer):
    def __init__(self, ga, E, n_in, n_out, bias):
        super().__init__(ga)
        self.weight = nn.Parameter(torch.empty(E, n_out, n_in, ga.G))
        self.bias = nn.Parameter(torch.empty(E, 1, n_out, 1)) if bias else None

    def forward(self, x):
        y = torch.einsum("encd,eocd->enod", x, self.ga.expand(self.weight))
        if self.bias is not None:
            y = y + self.bias * self.ga.e0
        return y


class EnsNormalization(_Layer):
    def __init__(self, ga, E, C):
        super().__init__(ga)
        self.a = nn.Parameter(torch.empty(E, C, ga.G))

    def forward(self, x):
        n = (self.ga.q(x) ** 2 + 1e-16) ** 0.25
        s = torch.sigmoid(self.a)[:, None]
        n = s * (n - 1) + 1
        return x / (self.ga.expand(n) + 1e-6)


class EnsMVSiLU(_Layer):
    def __init__(self, ga, E, C):
        super().__init__(ga)
        self.a = nn.Parameter(torch.empty(E, 1, C, ga.G))
        self.b = nn.Parameter(torch.empty(E, 1, C, ga.G))

    def forward(self, x):
        inv = torch.cat([x[..., :1], self.ga.q(x)[..., 1:]], -1)
        return self.ga.expand(torch.sigmoid(self.a * inv + self.b)) * x


class EnsFCGP(_Layer):
    def __init__(self, ga, E, n_in, n_out):
        super().__init__(ga)
        self.normalization = EnsNormalization(ga, E, n_in)
        self.linear_right = EnsMVLinear(ga, E, n_in, n_in, bias=False)
        self.linear_left = EnsMVLinear(ga, E, n_in, n_out, bias=True)
        self.weight = nn.Parameter(torch.empty(E, n_out, n_in, ga.P))

    def forward(self, x):
        xr = self.normalization(self.linear_right(x))
        t = self.ga.pairs(x, xr)
        return (self.linear_left(x) + torch.einsum("encpj,eocp->enoj", t, self.weight)) / math.sqrt(2)


class EnsGP(_Layer):
    def __init__(self, ga, E, C):
        super().__init__(ga)
        self.normalization = EnsNormalization(ga, E, C)
        self.linear_right = EnsMVLinear(ga, E, C, C, bias=False)
        self.linear_left = EnsMVLinear(ga, E, C, C, bias=True)
        self.weight = nn.Parameter(torch.empty(E, C, ga.P))

    def forward(self, x):
        xr = self.normalization(self.linear_right(x))
        t = self.ga.pairs(x, xr)
        return (self.linear_left(x) + torch.einsum("encpj,ecp->encj", t, self.weight)) / math.sqrt(2)


class EnsTralalero(nn.Module):
    def __init__(self, ga, E, widths, n_in, n_out=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = n_in
        for w in widths:
            self.blocks.append(nn.ModuleDict({"fc": EnsFCGP(ga, E, prev, w), "act1": EnsMVSiLU(ga, E, w),
                                              "gp": EnsGP(ga, E, w), "act2": EnsMVSiLU(ga, E, w)}))
            prev = w
        self.out = EnsFCGP(ga, E, prev, n_out)

    def forward(self, x):
        for b in self.blocks:
            x = b["act2"](b["gp"](b["act1"](b["fc"](x))))
        return self.out(x)


def _stack_into(module, refs):
    ref_params = [dict(r.named_parameters()) for r in refs]
    names = {n for n, _ in module.named_parameters()}
    assert names == set(ref_params[0]), sorted(names ^ set(ref_params[0]))
    with torch.no_grad():
        for n, p in module.named_parameters():
            src = torch.stack([rp[n].detach() for rp in ref_params])
            assert src.shape == p.shape, (n, src.shape, p.shape)
            p.copy_(src)


class EnsCGENN(nn.Module):
    """E copies of spectral_bias_sphere.CGENN: inputs [x (vector), e1, e2, e3 (constant frame)] as four multivectors."""

    def __init__(self, refs):
        super().__init__()
        r0 = refs[0]
        self.ga = _GA(r0.algebra)
        widths = [b["fc"].out_features for b in r0.net.blocks]
        self.net = EnsTralalero(self.ga, len(refs), widths, n_in=4)
        D = self.ga.D
        P = torch.zeros(3, D)
        for k in range(3):
            P[k, 1 + k] = 1.0                                   # blades 1, 2, 3 are e1, e2, e3
        self.register_buffer("P", P)
        self.register_buffer("frame", P.clone().unsqueeze(0))   # (1, 3, D): e1, e2, e3
        _stack_into(self, refs)

    def forward(self, x):                                       # x (E, N, 3) -> (E, N)
        vec = (x @ self.P).unsqueeze(2)                         # (E, N, 1, D)
        mv = torch.cat([vec, self.frame.unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)], 2)
        return self.net(mv)[..., 0, 0]


class EnsMLP(nn.Module):
    def __init__(self, refs):
        super().__init__()
        lins = [[m for m in r.net if isinstance(m, nn.Linear)] for r in refs]
        self.W = nn.ParameterList([nn.Parameter(torch.stack([l[i].weight.detach().T for l in lins])) for i in range(len(lins[0]))])
        self.b = nn.ParameterList([nn.Parameter(torch.stack([l[i].bias.detach()[None] for l in lins])) for i in range(len(lins[0]))])

    def forward(self, x):                                       # x (E, N, 3) -> (E, N)
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, W)
            if i < len(self.W) - 1:
                h = F.relu(h)
        return h[..., 0]


def build_ensemble(name, seeds):
    refs = []
    for sd in seeds:
        torch.manual_seed(sd)
        refs.append(S.build(name))
    return (EnsCGENN if name == "cgenn" else EnsMLP)(refs)


class EnsAdam:
    """torch.optim.Adam (betas 0.9, 0.999, eps 1e-8), elementwise, one lr per ensemble member; state on the device (CUDA-graph safe)."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.b1, self.b2, self.eps = params, betas[0], betas[1], eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = torch.zeros((), device=params[0].device, dtype=params[0].dtype)
        self.lr = [lr.view(-1, *[1] * (p.dim() - 1)) for p in params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - torch.pow(self.b1, self.t)
        bc2s = torch.sqrt(1 - torch.pow(self.b2, self.t))
        for p, m, v, lr in zip(self.params, self.m, self.v, self.lr):
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.sub_((lr / bc1) * m / (v.sqrt() / bc2s + self.eps))

    @torch.no_grad()
    def reset(self):
        for s in self.m + self.v:
            s.zero_()
        self.t.zero_()


# ------------------------------------------------------------------------------------------------------------------ experiment
def train_ensemble(name, members, args, device):
    """members: [(lr, seed)]. Returns one result dict per member."""
    E, L, LMAX = len(members), args.L, args.LMAX
    dev = torch.device(device)
    ens = build_ensemble(name, [s for _, s in members]).to(dev)
    xt = S.fibonacci_sphere(args.ntrain, seed=100)
    xe = S.fibonacci_sphere(args.neval, seed=200)
    tg = [S.make_target(s, L, [1.0] * L) for _, s in members]
    yt = torch.stack([S.target_fn(xt, b, a, L) for b, a in tg]).to(dev)
    ye = torch.stack([S.target_fn(xe, b, a, L) for b, a in tg]).to(dev)
    coef_true = torch.zeros(E, (LMAX + 1) ** 2, dtype=torch.float64)                # true SH coefficients, degrees 0..LMAX
    for e, (b, a) in enumerate(tg):
        o = 1
        for l in range(1, L + 1):
            coef_true[e, o:o + 2 * l + 1] = a[l - 1] * b[l - 1]
            o += 2 * l + 1
    pinv = torch.linalg.pinv(S.harmonics(xe, 0, LMAX)).to(dev)                      # (K, Ne) float64
    coef_true = coef_true.to(dev)
    xt_e = xt.to(dev).unsqueeze(0).expand(E, -1, -1).contiguous()
    xe_e = xe.to(dev).unsqueeze(0).expand(E, -1, -1).contiguous()
    lr = torch.tensor([l for l, _ in members], device=dev)
    params = list(ens.parameters())
    opt = EnsAdam(params, lr)
    starts = np.cumsum([0] + [2 * l + 1 for l in range(0, L + 1)])                  # block offsets of degrees 0..L

    def step():
        pred = ens(xt_e)
        loss = (pred - yt).pow(2).mean(-1)
        loss.sum().backward()
        opt.step()
        return loss

    graph = dev.type == "cuda" and not args.no_graph
    if graph:
        snap = [p.detach().clone() for p in params]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                for p in params:
                    p.grad = None
                step()
        torch.cuda.current_stream().wait_stream(side)
        with torch.no_grad():
            for p, s in zip(params, snap):
                p.copy_(s)
        opt.reset()
        for p in params:
            p.grad = None
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_loss = step()

    n_steps = args.steps
    if graph and args.budget_s:                                # time the step, then cut the run to the time budget
        snap2 = [p.detach().clone() for p in params]
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        tb = time.time()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        ms = (time.time() - tb) / 100
        with torch.no_grad():
            for p, s0 in zip(params, snap2):
                p.copy_(s0)
        opt.reset()
        n_steps = max(1000, min(args.steps, int(args.budget_s / ms) // 100 * 100))
        print(f"[{name}] {1e3 * ms:.1f} ms/step for E={E}: running {n_steps} steps (budget {args.budget_s:.0f}s)", flush=True)

    ckpt = sorted(set(int(round(s)) for s in np.geomspace(1, n_steps, args.n_ckpt)) | {0})
    rec = {"steps": [], "rho": [], "train_mse": [], "test_mse": []}

    def record(step_no):
        with torch.no_grad():
            fe = ens(xe_e).double()                                                  # (E, Ne)
            coef = fe @ pinv.T                                                       # (E, K) least-squares SH coefficients
            rho = []
            for l in range(1, L + 1):
                sl = slice(int(starts[l]), int(starts[l + 1]))
                rho.append((coef[:, sl] * coef_true[:, sl]).sum(1) / (coef_true[:, sl] ** 2).sum(1))
            rec["steps"].append(step_no)
            rec["rho"].append(torch.stack(rho, 1).float().cpu())                      # (E, L)
            rec["train_mse"].append((ens(xt_e) - yt).pow(2).mean(-1).float().cpu())
            rec["test_mse"].append((fe.float() - ye).pow(2).mean(-1).cpu())

    t0 = time.time()
    for s in range(n_steps + 1):
        if s in ckpt:
            record(s)
        if s == n_steps:
            break
        if graph:
            g.replay()
        else:
            for p in params:
                p.grad = None
            step()
        if args.log_every and (s + 1) % args.log_every == 0:
            el = time.time() - t0
            rho = rec["rho"][-1]
            msg = []
            for l in sorted({m[0] for m in members}):
                idx = [e for e, m in enumerate(members) if m[0] == l]
                msg.append(f"lr {l:g}: test mse {float(rec['test_mse'][-1][idx].mean()):.3g}, rho {' '.join(f'{v:.2f}' for v in rho[idx].mean(0))}")
            print(f"[{name}] step {s + 1}/{n_steps} {el:.0f}s ({1e3 * el / (s + 1):.1f} ms/step, last checkpoint {rec['steps'][-1]}) | " + " | ".join(msg), flush=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    rho = torch.stack(rec["rho"])                                                     # (C, E, L)
    out = []
    for e, (l, sd) in enumerate(members):
        out.append(dict(model=name, lr=l, seed=sd, n_steps=n_steps, params=sum(p[0].numel() for p in params), steps=rec["steps"],
                        rho=rho[:, e].tolist(), train_mse=[float(x[e]) for x in rec["train_mse"]],
                        test_mse=[float(x[e]) for x in rec["test_mse"]], ensemble_seconds=seconds, ensemble_size=E))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,mlp256,cgenn")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--LMAX", type=int, default=11)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--neval", type=int, default=2000)
    ap.add_argument("--n_ckpt", type=int, default=80)
    ap.add_argument("--log_every", type=int, default=2500)
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--budget_s", type=float, default=0.0, help="cut the steps so one model's ensemble takes about this long")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results_ens.json")
    args = ap.parse_args()
    members = [(l, s) for l in args.lrs for s in range(args.seeds)]
    res = {"args": vars(args)}
    for name in args.models.split(","):
        res[name] = train_ensemble(name, members, args, args.device)
        print(f"[{name}] done: {len(members)} runs in {res[name][0]['ensemble_seconds']:.0f}s, {res[name][0]['params']:,} params per run", flush=True)
        json.dump(res, open(args.out, "w"))
