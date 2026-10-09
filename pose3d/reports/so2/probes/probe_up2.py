"""probe_up2.py: how good can a C4 canonicaliser get on these features? (5-fold CV over all 1309 val images)

Features: pooled ResNet-101 features of the 4 exact rot90 views (cached to probe_up_feats.npy).
Canonicalisers, all argmax over the turn j of the rotated image:
  bin-lr    : binary 'upright' logistic regression on standardised features (as probe_up, but standardised)
  rot4-lr   : 4-way RotNet (which turn is this view?), joint log-likelihood sum_k log p(view k has turn (j-k)%4)
  rot4-mlp  : same with a 1-hidden-layer MLP
Transductive in the sense that the folds are val images (f0 never trained on them); no pose labels used.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

S = "/tmp/claude-1000/-home-bebra-GA-Research/79577114-4a69-4d0d-932d-b82986259792/scratchpad"
FEATS = f"{S}/probe_up_feats.npy"
torch.set_num_threads(20)

if not os.path.exists(FEATS):
    sys.path.insert(0, "/home/bebra/GA-Research/.claude/worktrees/flow-x1-prediction/pose3d")
    from pose3d.datasets.cache import InMemoryDataset
    from pose3d.evaluate import build_model
    val = InMemoryDataset.load(f"{S}/ckpt/val/pascal_val.pt")
    imgs = torch.stack([val[i]["img"] for i in range(len(val))])
    model, _ = build_model(torch.load(f"{S}/ckpt/base/GA-Research/pose3d/clifford_flow_cgenn_warp_synth_ema_b64.pth",
                                      map_location="cpu", weights_only=False), "cpu")
    bb = model.eval().adapter.backbone
    feats = []
    with torch.no_grad():
        for k in range(4):
            xk = torch.rot90(imgs, k, dims=(2, 3))
            feats.append(torch.cat([bb(xk[i:i + 32]).mean((2, 3)) for i in range(0, len(imgs), 32)]).numpy())
            print("views", k, flush=True)
    np.save(FEATS, np.stack(feats))
feats = np.load(FEATS)            # (4, N, C): feats[k] = features of rot90(x, k)
n = feats.shape[1]


def canon_acc(logp_fn, te):
    # image rotated by j has views rot90(x, j - k) at slot k; logp_fn(F, t) = log p(turn t | features F)
    hits = []
    for j in range(4):
        scores = np.stack([sum(logp_fn(feats[(jj - k) % 4, te], (jj - k) % 4) for k in range(4))
                           for jj in range(4)], 1) if False else None
        # candidate jj: assume view k has turn (jj - k) % 4; true views have turn (j - k) % 4
        cand = []
        for jj in range(4):
            cand.append(sum(logp_fn(feats[(j - k) % 4, te], (jj - k) % 4) for k in range(4)))
        hits.append(np.stack(cand, 1).argmax(1) == j)
    return np.concatenate(hits)


res = {}
kf = KFold(5, shuffle=True, random_state=0)
for name in ("bin-lr", "rot4-lr", "rot4-mlp"):
    hits = []
    for tr, te in kf.split(np.arange(n)):
        sc = StandardScaler().fit(feats[:, tr].reshape(-1, feats.shape[-1]))
        X = sc.transform(feats[:, tr].reshape(-1, feats.shape[-1]))
        y4 = np.repeat(np.arange(4), len(tr))
        if name == "bin-lr":
            clf = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced").fit(X, (y4 == 0).astype(int))
            def logp(F, t, clf=clf, sc=sc):
                lp = clf.predict_log_proba(sc.transform(F))
                return lp[:, 1] if t == 0 else lp[:, 0] / 3   # only the upright slot is informative
        else:
            clf = (LogisticRegression(C=0.01, max_iter=5000) if name == "rot4-lr"
                   else MLPClassifier((512,), alpha=1e-2, max_iter=300, early_stopping=True, random_state=0)).fit(X, y4)
            def logp(F, t, clf=clf, sc=sc):
                return np.log(clf.predict_proba(sc.transform(F))[:, t] + 1e-12)
        hits.append(canon_acc(logp, te))
        print(name, "fold", len(hits), float(hits[-1].mean()), flush=True)
    h = np.concatenate(hits)
    res[name] = dict(acc=float(h.mean()), n=int(h.size), errors=int((~h).sum()))
    print(name, res[name], flush=True)
json.dump(res, open(f"{S}/probe_up2.json", "w"), indent=1)
print(json.dumps(res, indent=1))
