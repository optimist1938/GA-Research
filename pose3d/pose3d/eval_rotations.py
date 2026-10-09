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
import math

import numpy as np
import torch
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader, Subset

from pose3d.engine.checkpoint import get_available_device
from pose3d.engine.metrics import acc_at, macro_metrics
from pose3d.evaluate import build_model
from pose3d.geometry.flow import rotor_multiply
from pose3d.geometry.rotor import matrix_to_rotor
from pose3d.models.c4_canon import C4Canonicalized
from pose3d.models.clifford_flow import RollPrior
from pose3d.models.c4_lift import rot90


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--path_to_datasets", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--eval_samples", type=int, default=32)
    # In-plane turns of the test image, degrees counter-clockwise on screen. Multiples of 90 are
    # exact (the C4 the variants are equivariant to); the others probe what C4 does not cover.
    # With --canonicalize the 90/180/270 rows equal the 0 row by construction (same canonical input).
    p.add_argument("--angles", type=int, nargs="+", default=[0, 90, 180, 270])
    # Couple the sampling noise to the turn (r0 -> g r0): an exactly equivariant model then gives
    # the same error image by image, not only in distribution.
    p.add_argument("--couple_noise", action="store_true")
    # Reweight the samples by a von Mises 'upright camera' prior fitted on train labels before the
    # medoid (RollPrior). Breaks the equivariance on purpose; report it next to the prior-free run.
    p.add_argument("--roll_prior", action="store_true")
    p.add_argument("--prior_train_images", type=int, default=4000)
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

    def fit(f):
        x = f.reshape(-1, f.shape[-1])
        y = np.tile((np.arange(4) == 0).astype(int), len(f))
        return LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(x, y)

    def acc(clf, f):
        return float((clf.decision_function(f.reshape(-1, f.shape[-1])).reshape(-1, 4).argmax(1) == 0).mean())

    n_hold = len(feats) // 5                       # held-out accuracy within train, then refit on all
    held = acc(fit(feats[n_hold:]), feats[:n_hold])
    clf = fit(feats)
    print(f"scorer: {len(feats)} train images x 4 turns, held-out (train split) canonicalisation acc {held:.4f}")
    wrapper.scorer.weight.copy_(torch.as_tensor(clf.coef_, dtype=torch.float32).view(1, -1))
    wrapper.scorer.bias.fill_(float(clf.intercept_[0]))
    return {"n_train_images": len(feats), "heldout_acc": held, "insample_acc": acc(clf, feats)}


def r_z(deg):
    a = math.radians(deg)
    return torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def turn_image(img, deg):
    """Counter-clockwise on screen by `deg`. Multiples of 90 are exact pixel permutations (rot90);
    other angles are bilinear about the image centre with zero fill (the warp's own border)."""
    if deg % 90 == 0:
        return rot90(img, (deg // 90) % 4)
    import torchvision.transforms.functional as TF
    return TF.rotate(img, deg, interpolation=TF.InterpolationMode.BILINEAR)


@torch.no_grad()
def main():
    args = parse()
    device = get_available_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    flow, _ = build_model(checkpoint, device)
    flow.eval()
    model = flow
    result = {"checkpoint": args.checkpoint, "canonicalize": args.canonicalize, "couple_noise": args.couple_noise,
              "steps": args.steps, "eval_samples": args.eval_samples, "angles": {}}
    if args.canonicalize:
        model = C4Canonicalized(flow).to(device).eval()
        result["scorer"] = fit_scorer(model, args.path_to_datasets, args.canon_train_images,
                                      args.batch_size, args.num_workers, device, args.seed)

    prior = None
    if args.roll_prior:
        train = Pascal3D(args.path_to_datasets, train=True, use_warp=False, use_synth=False)
        idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.prior_train_images]
        prior = RollPrior.fit(torch.stack([train[int(i)]["rot"] for i in idx]))
        result["roll_prior"] = prior.state()
        print("roll prior:", prior.state(), flush=True)

    test = Pascal3D(args.path_to_datasets, train=False)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    per_sample = {}
    for deg in args.angles:
        # Turning the image counter-clockwise by deg maps the label R to R_z(-deg) R (G_TURN for 90).
        g = r_z(-deg).to(device)
        g_rotor = matrix_to_rotor(g.unsqueeze(0))
        gen = torch.Generator(device="cpu").manual_seed(args.seed)    # same noise draws at every angle
        errs, classes, turns = [], [], []
        for batch in loader:
            img, rot = batch["img"].to(device), batch["rot"].to(device).float()
            cls = batch["cls"].to(device)
            r0 = torch.randn(img.shape[0] * args.eval_samples, 4, generator=gen)
            r0 = (r0 / r0.norm(dim=-1, keepdim=True)).to(device)
            if args.couple_noise and not args.canonicalize:
                # equivariance maps the noise with the image: r0 -> g r0 (Lean: euler_equivariant)
                r0 = rotor_multiply(g_rotor.expand_as(r0), r0, flow.algebra)
            pred = model.predict(turn_image(img, deg), cls, n_samples=args.eval_samples, steps=args.steps,
                                 noise=r0, roll_prior=prior)
            errs.append(geodesic_deg(pred, g @ rot).cpu())
            classes.append(batch["cls"].view(-1))
            if args.canonicalize:
                turns.append(model.last_turn.cpu())
        err, cls = torch.cat(errs).numpy(), torch.cat(classes).numpy()
        macro = macro_metrics(err, cls)
        row = dict(median=float(np.median(err)), class_mean_median=macro["class_mean_median_error"],
                   mean=float(err.mean()), acc15=acc_at(err, 15), acc30=acc_at(err, 30),
                   class_medians={int(c): float(m) for c, m in macro["class_medians"].items()})
        per_sample[f"err_{deg}"], per_sample["cls"] = err, cls
        if args.canonicalize:
            c = torch.cat(turns).numpy()
            per_sample[f"turn_{deg}"] = c
            if deg % 90 == 0:
                hit = c == (deg // 90) % 4
                row["canonicaliser_acc"] = float(hit.mean())
                row["canonicaliser_acc_per_class"] = {int(k): float(hit[cls == k].mean()) for k in np.unique(cls)}
        result["angles"][deg] = row
        print(f"angle {deg:>4} deg: " + ", ".join(f"{n} {v:.4f}" for n, v in row.items()
                                                 if isinstance(v, float)), flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=1)
        np.savez(args.output_json.replace(".json", "") + "_per_sample.npz", **per_sample)


if __name__ == "__main__":
    main()
