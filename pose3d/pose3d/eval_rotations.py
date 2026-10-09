"""Test-set metrics under 90-degree turns of the image (the C4 subgroup of the in-plane SO(2)).

    python -m pose3d.eval_rotations --checkpoint ckpt.pth --path_to_datasets <pascal3d>
        [--canonicalize] [--canon_train_images 4000] [--steps 20] [--eval_samples 32]
        [--output_json out.json]

For every turn k in 0..3 the test image is rot90(x, k) (counter-clockwise on screen) and the label
becomes G^k R with G = R_z(-90 deg) (so2_head.py's convention). An exactly C4-equivariant model
scores the same at every k (up to its sampling noise); the 8.95 deg recipe collapses at k = 1, 2, 3.

--canonicalize wraps the checkpoint in variant B (models/c4_canon.py): a logistic-regression
'uprightness' scorer is fitted on the checkpoint's own pooled backbone features of upright train
images (label 1) and of their three turns (label 0), then c(x) = argmax_k score(rot90(x, -k)).
Only train images are used to fit it.
"""

import argparse
import json

import numpy as np
import torch
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader, Subset

from pose3d.engine.checkpoint import get_available_device
from pose3d.engine.metrics import acc_at, macro_metrics
from pose3d.evaluate import build_model
from pose3d.models.c4_canon import G_TURN, C4Canonicalized
from pose3d.models.c4_lift import rot90


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--path_to_datasets", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eval_samples", type=int, default=32)
    p.add_argument("--turns", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--canonicalize", action="store_true")
    p.add_argument("--canon_train_images", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def geodesic_deg(a, b):
    tr = torch.einsum("bij,bij->b", a, b)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


@torch.no_grad()
def fit_scorer(wrapper, path, n_images, batch_size, num_workers, device, seed):
    from sklearn.linear_model import LogisticRegression

    train = Pascal3D(path, train=True, use_warp=False, use_synth=False)
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:n_images]
    loader = DataLoader(Subset(train, idx.tolist()), batch_size=batch_size, num_workers=num_workers)
    feats = []
    for batch in loader:
        feats.append(wrapper.view_features(batch["img"].to(device)).cpu())   # (B, 4, C)
    feats = torch.cat(feats).numpy()
    x = feats.reshape(-1, feats.shape[-1])
    y = np.tile((np.arange(4) == 0).astype(int), len(feats))
    clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(x, y)
    turn_acc = float((clf.decision_function(x).reshape(-1, 4).argmax(1) == 0).mean())
    print(f"scorer fitted on {len(feats)} train images x 4 turns, train canonicalisation acc {turn_acc:.4f}")
    with torch.no_grad():
        wrapper.scorer.weight.copy_(torch.as_tensor(clf.coef_, dtype=torch.float32).view(1, -1))
        wrapper.scorer.bias.fill_(float(clf.intercept_[0]))
    return turn_acc


@torch.no_grad()
def main():
    args = parse()
    device = get_available_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_model(checkpoint, device)
    model.eval()
    result = {"checkpoint": args.checkpoint, "canonicalize": args.canonicalize,
              "steps": args.steps, "eval_samples": args.eval_samples, "turns": {}}
    if args.canonicalize:
        model = C4Canonicalized(model).to(device).eval()
        result["scorer_train_acc"] = fit_scorer(model, args.path_to_datasets, args.canon_train_images,
                                                args.batch_size, args.num_workers, device, args.seed)

    test = Pascal3D(args.path_to_datasets, train=False)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    for k in args.turns:
        g = torch.linalg.matrix_power(G_TURN, k).to(device)
        torch.manual_seed(args.seed)   # same draws at every turn
        errs, classes, turns_hit = [], [], []
        for batch in loader:
            img, rot = batch["img"].to(device), batch["rot"].to(device).float()
            cls = batch.get("cls")
            x = rot90(img, k)
            pred = model.predict(x, cls.to(device) if cls is not None else None,
                                 n_samples=args.eval_samples, steps=args.steps)
            errs.append(geodesic_deg(pred, g @ rot).cpu())
            classes.append((cls if cls is not None else batch["cls_eval"]).view(-1))
            if args.canonicalize:
                turns_hit.append((model.canonical_turn(x).cpu() == k).float())
        err, cls = torch.cat(errs).numpy(), torch.cat(classes).numpy()
        macro = macro_metrics(err, cls)
        row = dict(median=float(np.median(err)), class_mean_median=macro["class_mean_median_error"],
                   mean=float(err.mean()), acc15=acc_at(err, 15), acc30=acc_at(err, 30))
        if args.canonicalize:
            row["canonicaliser_acc"] = float(torch.cat(turns_hit).mean())
        result["turns"][k] = row
        print(f"turn {k * 90:>3} deg: " + ", ".join(f"{n} {v:.4f}" for n, v in row.items()), flush=True)
    if args.output_json:
        json.dump(result, open(args.output_json, "w"), indent=1)


if __name__ == "__main__":
    main()
