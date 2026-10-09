"""stage0.py: offline estimate of what exact C4 equivariance costs on the upright test (no GPU).

An exactly C4-equivariant model must find 'up' from content. Where content does not tell (the
canonicaliser fails), its posterior has the turned modes G^m R as well; the medoid may land on one.
Model: baseline per-image prediction P (CGENN 7.88 recipe, 8 samples x 10 steps); for images where the
5-fold CV canonicaliser picks turn m != 0, the equivariant model's prediction is G^m P with
probability q (q = 1: it always follows the wrong turn, the pessimistic bound; q = 0.5: a coin flip
between the right and the wrong mode). With the roll prior (RollPrior fitted on the other folds'
labels) the turned modes get weight ~1e-10 and the cost vanishes unless the prior itself is wrong:
reported as the fraction of images whose upright mode loses to a turned one under the prior.
"""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/bebra/GA-Research/.claude/worktrees/flow-x1-prediction/pose3d")
from pose3d.datasets.cache import InMemoryDataset
from pose3d.evaluate import build_model
from pose3d.models.c4_canon import G_TURN
from pose3d.models.clifford_flow import RollPrior

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
torch.set_num_threads(20)
feats = np.load(f"{S}/probe_up_feats.npy")      # (4, N, C), feats[k] = rot90(x, k)
val = InMemoryDataset.load(f"{S}/ckpt/val/pascal_val.pt")
n = len(val)
rots = torch.stack([val[i]["rot"] for i in range(n)]).float()

pred_path = f"{S}/stage0_pred.pt"
try:
    pred = torch.load(pred_path)
except FileNotFoundError:
    imgs = torch.stack([val[i]["img"] for i in range(n)])
    model, _ = build_model(torch.load(f"{S}/ckpt/base/GA-Research/pose3d/clifford_flow_cgenn_warp_synth_ema_b64.pth",
                                      map_location="cpu", weights_only=False), "cpu")
    model.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        pred = torch.cat([model.predict(imgs[i:i + 32], n_samples=8, steps=10) for i in range(0, n, 32)])
    torch.save(pred, pred_path)


def geo(a, b):
    tr = torch.einsum("bij,bij->b", a, b)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).numpy()


# 5-fold CV canonicaliser on the upright images: 4-way RotNet logistic regression, joint likelihood
turn = np.zeros(n, dtype=int)
prior_fail = np.zeros(n, dtype=bool)
for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
    sc = StandardScaler().fit(feats[:, tr].reshape(-1, feats.shape[-1]))
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(
        sc.transform(feats[:, tr].reshape(-1, feats.shape[-1])), np.repeat(np.arange(4), len(tr)))
    lp = [np.log(clf.predict_proba(sc.transform(feats[k, te])) + 1e-12) for k in range(4)]
    # feats[k] = rot90(y, k) of the observed (upright) image y. Candidate jj = "y is the upright image
    # turned by jj", under which rot90(y, k) has turn (jj + k) % 4.
    cand = np.stack([sum(lp[k][:, (jj + k) % 4] for k in range(4)) for jj in range(4)], 1)
    turn[te] = cand.argmax(1)
    prior = RollPrior.fit(rots[tr])
    w = torch.stack([prior.weights(torch.linalg.matrix_power(G_TURN, m) @ rots[te]) for m in range(4)], 1)
    prior_fail[te] = (w[:, 1:].max(1).values > w[:, 0]).numpy()

err0 = geo(pred, rots)
res = {"n": n, "canonicaliser_acc_cv": float((turn == 0).mean()),
       "baseline": dict(median=float(np.median(err0)), acc15=float((err0 < 15).mean()), acc30=float((err0 < 30).mean())),
       "prior_picks_wrong_turn_on_labels": float(prior_fail.mean())}
rng = np.random.default_rng(0)
for q in (1.0, 0.5):
    errs = []
    for rep in range(20 if q < 1 else 1):
        follow = (turn != 0) & (rng.random(n) < q)
        g = torch.stack([torch.linalg.matrix_power(G_TURN, int(m)) for m in turn])
        p2 = torch.where(torch.as_tensor(follow).view(-1, 1, 1), g @ pred, pred)
        errs.append(geo(p2, rots))
    e = np.stack(errs)
    res[f"equivariant_no_prior_q{q}"] = dict(median=float(np.median(e, 1).mean()),
                                             acc15=float((e < 15).mean()), acc30=float((e < 30).mean()))
print(json.dumps(res, indent=1))
json.dump(res, open(f"{S}/stage0.json", "w"), indent=1)
