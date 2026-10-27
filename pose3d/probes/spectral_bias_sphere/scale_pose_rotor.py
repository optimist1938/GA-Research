"""Pose-dependent spectral bias with the pose given as a ROTOR (GATr's embedding of a rotation), instead of the frame (Re1)e123, Re2, Re3.
cgenn_rotor : CGENN on 2 multivector channels [x as vector, rotor (1, e12, e13, e23)];
gatr_rotor  : GATr, 2 tokens: pose = gatr.interface.embed_rotation(rotor), point = x as grade-1 vector;
gatr_recipe : GATr, 2 tokens: pose = embed_rotation(rotor), point = gatr.interface.embed_point(x) (trivector): the GATr paper's embeddings.
One seed, equal amplitudes, base size (~98k), per-model time cap."""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import spectral_bias_pose as P
from pose3d.geometry.rotor import matrix_to_rotor, embed_rotor
from gatr.interface import embed_point, embed_rotation


def quat_ijkw(R):
    """Rotor (1, e12, e13, e23) of R as an ijkw quaternion such that embed_rotation puts it in blades (0, 8, 9, 10) unchanged."""
    r = matrix_to_rotor(R)
    return torch.stack([-r[:, 3], r[:, 2], -r[:, 1], r[:, 0]], -1)


class CGENNRotor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.algebra = P.CliffordAlgebra((1, 1, 1))
        self.net = P.TralaleroTralala(self.algebra, in_features=2, hidden_dim=[hidden], out_features=1)

    def forward(self, x, R):
        mv = torch.cat([self.algebra.embed(x, (1, 2, 3)).unsqueeze(1), embed_rotor(matrix_to_rotor(R), self.algebra).unsqueeze(1)], 1)   # (B, 2, 8)
        return self.net(mv)[:, 0, 0]


class GATrRotor(nn.Module):
    def __init__(self, mv, sc, blocks, point_as_trivector, heads=4):
        super().__init__()
        GATr = P._import_gatr()
        self.net = GATr(in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=mv, in_s_channels=2, out_s_channels=1, hidden_s_channels=sc,
                        attention={"num_heads": heads}, mlp={}, num_blocks=blocks)
        self.register_buffer("_idx", torch.tensor(P._PGA_INDEX))
        self.register_buffer("_eye", torch.eye(2))
        self.tri = point_as_trivector

    def forward(self, x, R):
        b = x.shape[0]
        pose = embed_rotation(quat_ijkw(R))                                   # (B, 16)
        if self.tri:
            pt = embed_point(x)
        else:
            pt = x.new_zeros(b, 16)
            pt[:, self._idx[1:4]] = x                                         # grade-1 e1..e3
        mv = torch.stack([pose, pt], 1).unsqueeze(2)                           # (B, 2 tokens, 1 channel, 16)
        _, out_s = self.net(mv, scalars=self._eye.to(x.dtype).expand(b, -1, -1))
        return out_s[:, 0, 0]


MODELS = {"cgenn_rotor": lambda: CGENNRotor(84), "gatr_rotor": lambda: GATrRotor(8, 32, 4, False), "gatr_recipe": lambda: GATrRotor(8, 32, 4, True)}
P.build = lambda name: MODELS[name]()

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="cgenn_rotor,gatr_rotor,gatr_recipe")
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
ap.add_argument("--out", default="results_pose_rotor.json")
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
