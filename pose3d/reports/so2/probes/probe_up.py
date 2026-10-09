"""probe_up.py: can the in-plane 'up' be read from image content with the pose-trained ResNet-101?

An exactly C4-equivariant model has no absolute image axis, so it must find the scene's up direction
from content. Here: features of the 4 exact rot90 copies of each (upright) validation image from the
fine-tuned backbone of the CGENN baseline; a linear 'uprightness' scorer s is trained on
view k = 0 vs k = 1, 2, 3 (train split), and the equivariant canonicaliser
c(x) = argmax_k s(rot90(x, -k)) is evaluated on held-out images that were first rotated by a
random j: it must return j. Also reports the C4 harmonics of the scorer: the phase of
c1 = sum_k i^-k s_k points to the up direction (that is what the C4-lift head can use).
"""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/home/bebra/GA-Research/.claude/worktrees/flow-x1-prediction/pose3d")
from pose3d.datasets.cache import InMemoryDataset
from pose3d.evaluate import build_model

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
CKPT = f"{S}/ckpt/base/GA-Research/pose3d/clifford_flow_cgenn_warp_synth_ema_b64.pth"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
torch.set_num_threads(20)

val = InMemoryDataset.load(f"{S}/ckpt/val/pascal_val.pt")
idx = torch.randperm(len(val), generator=torch.Generator().manual_seed(0))[:N]
imgs = torch.stack([val[int(i)]["img"] for i in idx])
model, _ = build_model(torch.load(CKPT, map_location="cpu", weights_only=False), "cpu")
model.eval()
bb = model.adapter.backbone

feats = []  # (4, N, 2048): pooled features of rot90(x, k) for k = 0..3
with torch.no_grad():
    for k in range(4):
        xk = torch.rot90(imgs, k, dims=(2, 3))
        feats.append(torch.cat([bb(xk[i:i + 32]).mean((2, 3)) for i in range(0, N, 32)]).numpy())
        print("views", k, flush=True)
feats = np.stack(feats)

n_tr = N * 3 // 4
Xtr = np.concatenate([feats[k, :n_tr] for k in range(4)])
ytr = np.concatenate([np.full(n_tr, int(k == 0)) for k in range(4)])
clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)


def score(f):
    return clf.decision_function(f)


# held-out: rotate by random j, canonicalise by argmax over the 4 views of the rotated image
rng = np.random.default_rng(0)
te = np.arange(n_tr, N)
j = rng.integers(0, 4, size=len(te))
# views of rot90(x, j) at rot90(., -k) are rot90(x, j - k): reuse feats[(j - k) % 4]
s = np.stack([score(feats[(j - k) % 4, te]) for k in range(4)], 1)          # (n_te, 4)
pred = s.argmax(1)
acc = float((pred == j).mean())
s_up = np.stack([score(feats[k, te]) for k in range(4)], 1)                  # unrotated: k=0 is up
c1 = (s_up * (1j) ** (-np.arange(4))).sum(1)
phase_ok = float((np.abs(np.angle(c1)) < np.pi / 4).mean())
res = dict(n_train=int(n_tr), n_test=int(len(te)), canonicaliser_acc=acc,
           c1_phase_points_up=phase_ok,
           upright_rank_top1=float((s_up.argmax(1) == 0).mean()))
print(json.dumps(res, indent=1))
json.dump(res, open(f"{S}/probe_up.json", "w"), indent=1)
