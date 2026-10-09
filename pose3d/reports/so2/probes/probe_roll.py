"""probe_roll.py: how far is a trained model from SO(2) equivariance, and how much of its error is roll?

1. Rotate each validation image in-plane by phi (torchvision rotate, counter-clockwise on screen,
   zero fill) and its label by R_z(sign * phi) for both signs; the sign with the lower error is the
   pipeline's convention (expected: CCW on screen -> R_z(-phi), see so2_head.py). The error vs phi
   curve measures how far the model is from equivariance (an exactly equivariant model is flat).
2. Swing-twist split of the unrotated error E = R_pred R_gt^T about the optical axis e3:
   twist = in-plane (roll) part, swing = out-of-plane part.
"""
import json
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, "/home/bebra/GA-Research/.claude/worktrees/flow-x1-prediction/pose3d")
from pose3d.datasets.cache import InMemoryDataset
from pose3d.evaluate import build_model

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
CKPT = f"{S}/ckpt/base/GA-Research/pose3d/clifford_flow_cgenn_warp_synth_ema_b64.pth"
N, SAMPLES, STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 384, 8, 10
PHIS = [0, 5, 10, 20, 45, 90, 180]
DEG = 180 / np.pi
torch.set_num_threads(20)
torch.manual_seed(0)


def rz(phi_deg):
    a = np.radians(phi_deg)
    return torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=torch.float32)


def geo(a, b):
    tr = torch.einsum("bij,bij->b", a, b)
    return torch.arccos(((tr - 1) / 2).clamp(-1, 1)) * DEG


def twist_swing(err):
    """err (B,3,3). Twist about e3 from the quaternion (w, z) part, swing = the rest."""
    w = 0.5 * torch.sqrt((1 + err[:, 0, 0] + err[:, 1, 1] + err[:, 2, 2]).clamp(min=1e-9))
    z = (err[:, 1, 0] - err[:, 0, 1]) / (4 * w)
    twist = 2 * torch.atan2(z.abs(), w.abs()) * DEG
    total = geo(err, torch.eye(3).expand_as(err))
    swing = 2 * torch.arccos((torch.sqrt(w**2 + z**2)).clamp(max=1)) * DEG
    return twist, swing, total


val = InMemoryDataset.load(f"{S}/ckpt/val/pascal_val.pt")
idx = torch.linspace(0, len(val) - 1, N).long()
imgs = torch.stack([val[int(i)]["img"] for i in idx])
rots = torch.stack([val[int(i)]["rot"] for i in idx]).float()
model, _ = build_model(torch.load(CKPT, map_location="cpu", weights_only=False), "cpu")
model.eval()

res = {}
with torch.no_grad():
    for phi in PHIS:
        torch.manual_seed(1)
        x = imgs if phi == 0 else TF.rotate(imgs, phi, interpolation=TF.InterpolationMode.BILINEAR)
        pred = torch.cat([model.predict(x[i:i + 32], n_samples=SAMPLES, steps=STEPS) for i in range(0, N, 32)])
        row = {}
        for sign in (+1, -1):
            gt = rz(sign * phi) @ rots
            row[sign] = float(geo(pred, gt).median())
        res[phi] = row
        if phi == 0:
            tw, sw, tot = twist_swing(pred @ rots.transpose(1, 2))
            res["split"] = dict(total_median=float(tot.median()), twist_median=float(tw.median()),
                                swing_median=float(sw.median()), twist_mean=float(tw.mean()),
                                swing_mean=float(sw.mean()),
                                frac_twist_dominant=float((tw > sw).float().mean()))
        print(phi, row, flush=True)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{S}/probe_roll.json", "w"), indent=1)
