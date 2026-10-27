"""Pose-dependent spectral bias with ~4x the parameters obtained through DEPTH (width unchanged): MLP width 138 x 24 hidden layers, CGENN 2 blocks of 84 and
3 blocks of 64, GATr 8 mv / 32 scalar channels x 16 blocks. One seed, equal amplitudes; each model's steps are cut to fit a time cap."""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import spectral_bias_pose as P

MODELS = {"mlp_d24": lambda: P.MLP(138, depth=24), "cgenn_2x84": lambda: P.CGENN([84] * 2), "cgenn_3x64": lambda: P.CGENN([64] * 3), "gatr_16b": lambda: P.GATrNet(8, 32, 16)}
P.build = lambda name: MODELS[name]()

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="mlp_d24,cgenn_2x84,cgenn_3x64,gatr_16b")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--cap_s", type=float, default=6000.0, help="per-model time cap: the step count is cut if the timed step projects beyond it")
ap.add_argument("--ntrain", type=int, default=4000)
ap.add_argument("--neval", type=int, default=3000)
ap.add_argument("--n_ckpt", type=int, default=40)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--LMAX", type=int, default=11)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--amps", default="equal")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="results_pose_depth.json")
args = ap.parse_args()
amp = [1.0] * args.L
res = {"args": vars(args), "amp": amp}
dev = torch.device(args.device)
for name in args.models.split(","):
    m = P.build(name).to(dev); xt, Rt, _ = P.sample(args.ntrain, 100 + args.seed); xt, Rt = xt.to(dev), Rt.to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3); ts = []
    for i in range(6):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time(); opt.zero_grad(); m(xt, Rt).pow(2).mean().backward(); opt.step()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ms = sum(ts[2:]) / 4; a = copy.copy(args); a.steps = int(max(500, min(args.steps, args.cap_s / ms) // 100 * 100)); del m, opt
    print(f"{name}: {1e3 * ms:.0f} ms/step -> {a.steps} steps (cap {args.cap_s:.0f}s)", flush=True)
    r = P.run(name, args.seed, a, amp); r["n_steps"] = a.steps
    res[name] = [r]
    print(f"{name} seed {args.seed}: params {r['params']}, {r['seconds']:.0f}s, steps {a.steps}, final train mse {r['train_mse'][-1]:.3f}, test mse {r['test_mse'][-1]:.3f}, "
          f"rho = {' '.join(f'{v:.2f}' for v in r['rho'][-1])}", flush=True)
    json.dump(res, open(args.out, "w"))
