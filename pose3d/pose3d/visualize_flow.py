"""Visualize the flow-matching trajectories of a trained CliffordFlow on SO(3): the path from noise to a pose.

For a few validation images the flow is integrated from several random rotations (the Euler loop of CliffordFlow.predict, but every
intermediate rotation is kept). Each rotation R_t is shown relative to the ground truth, R_gt^T R_t, as an axis-angle vector inside the
ball of radius pi: the true pose is the origin, the distance from the origin is the geodesic error, and a path that ends near the origin
has found the pose. Opposite points of the ball's surface are the same rotation, so a path can jump across it (drawn as a break).

    python -m pose3d.visualize_flow --checkpoint a.pth --label GATr --checkpoint b.pth --label MLP \
        --val_cache pascal_val.pt --out viz/

With --wandb the same visualizations are logged to Weights & Biases straight from the checkpoints, without any training: one run per
checkpoint (job type "visualize"), holding the predictions table, the paths figure, the film strip and the animation.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from pose3d.engine.flow_viz import (angle_deg, animation, errors, figure_compare, figure_filmstrip, figure_frames, figure_hopf, figure_paths,  # noqa: F401
                                    interactive_frames, pick_images, trajectories)
from pose3d.engine.rotor_viz import figure_rotors, interactive_rotors
from pose3d.evaluate import build_model


PASCAL_VAL_COUNTS = (228, 88, 130, 91, 125, 185, 96, 7, 90, 27, 100, 142)     # test images per class; the validation set is ordered by category


def render_files(models, imgs, rots, picks, errs, a, out, cls=None):
    """Write every visualization of every model into out/: frames_*.png, chart_*.png, filmstrip_*.png, flow_*.gif and one interactive.html."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    interactive_frames(models, imgs, rots, picks, min(a.k, 12), a.steps, a.seed, out / "interactive.html", cls=cls, errs=list(errs))
    for lab, m in models.items():
        e_m = errors(m, imgs[picks], rots[picks])
        figure_hopf(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"hopf_{lab}.png")
        figure_rotors(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"rotors_{lab}.png")
        figure_frames(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"frames_{lab}.png")
        figure_paths(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"chart_{lab}.png")
        figure_filmstrip(lab, m, imgs, rots, picks[1], a.steps, a.seed, out / f"filmstrip_{lab}.png")
        animation(lab, m, imgs, rots, picks[1], a.k, a.steps, a.seed, out / f"flow_{lab}.gif")
        print("done", lab, flush=True)
    interactive_rotors(models, imgs, rots, picks, min(a.k, 12), a.steps, a.seed, out / "rotors.html", cls=cls, errs=list(errs))
    return out


def log_to_wandb(models, a):
    """One W&B run per checkpoint with the final-style visualizations (table, paths, film strip, animation)."""
    import wandb
    from pose3d.datasets.cache import InMemoryDataset
    from pose3d.engine.flow_viz import wandb_media

    ds = InMemoryDataset.load(a.val_cache)
    if len(ds) == sum(PASCAL_VAL_COUNTS):
        ds.set_eval_classes(np.repeat(np.arange(12), PASCAL_VAL_COUNTS))        # class names in the table
    imgs, rots = torch.stack([ds[i]["img"] for i in range(0, len(ds), 14)]), np.stack([ds[i]["rot"].numpy() for i in range(0, len(ds), 14)])
    for (lab, model), ck in zip(models.items(), a.checkpoint):
        run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=f"{a.wandb_prefix}{lab}", job_type="visualize", group=a.wandb_group,
                         config=dict(checkpoint=Path(ck).name, label=lab, k=a.k, steps=a.steps, seed=a.seed, images=a.wandb_images))
        run.summary["viz_pool_median_error_deg"] = float(np.median(errors(model, imgs, rots)))
        run.log(wandb_media(model, ds, n=a.wandb_images, k=a.k, steps=a.steps, seed=a.seed, final=True))
        if a.wandb_artifact:                          # the full-resolution files, downloadable from the run
            picks, errs = pick_images(model, imgs, rots, min(a.n_pool, len(imgs)), a.quantiles)
            d = render_files({lab: model}, imgs, rots, picks, errs, a, Path(a.out) / lab)
            art = wandb.Artifact(f"flow-viz-{lab}", type="visualization", metadata=dict(checkpoint=Path(ck).name, k=a.k, steps=a.steps, seed=a.seed))
            art.add_dir(str(d)); run.log_artifact(art)
        print(f"logged {lab} to {run.url}", flush=True)
        run.finish()


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt, "cpu")
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--label", action="append", required=True)
    p.add_argument("--val_cache", required=True, help="pascal_val.pt of the Pascal3D RAM cache"); p.add_argument("--out", default="viz")
    p.add_argument("--n_pool", type=int, default=96, help="validation images scored to choose the examples")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.8, 0.95], help="error quantiles of the examples (of the first model)")
    p.add_argument("--wandb", action="store_true", help="log the visualizations to W&B from the checkpoints (no training)")
    p.add_argument("--wandb_project", default="3D Pose Estimation"); p.add_argument("--wandb_entity", default="clifforders")
    p.add_argument("--wandb_prefix", default="viz_"); p.add_argument("--wandb_group", default="visualizations"); p.add_argument("--wandb_images", type=int, default=8)
    p.add_argument("--wandb_artifact", action="store_true", help="also upload the PNG/GIF/interactive HTML files as a W&B artifact of each run")
    p.add_argument("--k", type=int, default=24); p.add_argument("--steps", type=int, default=20); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); assert len(a.checkpoint) == len(a.label)
    torch.set_num_threads(4); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(a.val_cache, map_location="cpu", weights_only=True)
    imgs, rots = blob["imgs"].float() / 255, blob["targets"].float().numpy()
    models = {lab: load(ck) for ck, lab in zip(a.checkpoint, a.label)}
    if a.wandb:
        log_to_wandb(models, a)
        return
    first = next(iter(models.values()))
    picks, errs = pick_images(first, imgs, rots, a.n_pool, a.quantiles)
    print("examples (val index, medoid error of", next(iter(models)), "):", list(zip(picks, [round(e, 1) for e in errs])), flush=True)
    cls = np.repeat(np.arange(12), PASCAL_VAL_COUNTS) if len(imgs) == sum(PASCAL_VAL_COUNTS) else None
    render_files(models, imgs, rots, picks, errs, a, out, cls=cls)
