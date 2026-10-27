"""Pose-dependent spectral bias at ~4x the parameters of the base models (one seed, equal amplitudes): MLP width 280, CGENN hidden 160, GATr 16/64 channels."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import spectral_bias_pose as P

P.build = lambda name: {"mlp_x4": lambda: P.MLP(280), "cgenn_x4": lambda: P.CGENN([160]), "gatr_x4": lambda: P.GATrNet(16, 64, 4)}[name]()

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="mlp_x4,cgenn_x4,gatr_x4")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--ntrain", type=int, default=4000)
ap.add_argument("--neval", type=int, default=3000)
ap.add_argument("--n_ckpt", type=int, default=40)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--LMAX", type=int, default=11)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--amps", default="equal")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="results_pose_x4.json")
args = ap.parse_args()
amp = [1.0] * args.L
res = {"args": vars(args), "amp": amp}
for name in args.models.split(","):
    r = P.run(name, args.seed, args, amp)
    res[name] = [r]
    print(f"{name} seed {args.seed}: params {r['params']}, {r['seconds']:.0f}s, final train mse {r['train_mse'][-1]:.3f}, test mse {r['test_mse'][-1]:.3f}, "
          f"rho = {' '.join(f'{v:.2f}' for v in r['rho'][-1])}", flush=True)
    json.dump(res, open(args.out, "w"))
