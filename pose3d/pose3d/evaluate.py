"""Re-score a saved CliffordFlow checkpoint with multi-sample prediction.

    python -m pose3d.evaluate --artifact <entity/project/name.pth:vN> --path_to_datasets ...
    python -m pose3d.evaluate --checkpoint <local .pth> --path_to_datasets ...

Loads the checkpoint (from W&B, or a local file) and runs one evaluation pass over the
Pascal3D test split at several sample counts, so the gain from mode selection (the geodesic
medoid) is measurable on the same weights. Besides the median over all test images it
reports the per-class medians and their mean, the number the IPDF / Image2Sphere / Rotation
Laplace tables use. Nothing is trained and no W&B run is created.

`--steps` repeats every pass at several Euler step counts (CliffordFlow only), and every pass is
timed: the seconds spent inside `predict` (CUDA-synchronised, data loading excluded), so the
accuracy / inference-time trade-off of fewer ODE steps is read off one table.

Checkpoints written before the nested config store a flat `config` dict; both layouts load.
CliffordFlow and i2s_real (the Image2Sphere baseline) checkpoints are supported.
"""

import argparse
import functools
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader

from pose3d.config import Config
from pose3d.engine.checkpoint import get_available_device
from pose3d.engine.metrics import acc_at, calculate_evaluation_metrics, macro_metrics
from pose3d.models.clifford_flow import CliffordFlow


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str,
                        default="clifforders/3D Pose Estimation/clifford_flow_pretrained.pth:v1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="local .pth to score instead of downloading --artifact")
    parser.add_argument("--output_json", type=str, default=None,
                        help="also write every number printed here to this file")
    parser.add_argument("--i2s_eval_rec_level", type=int, default=None,
                        help="i2s_real only: SO(3) HEALPix level of the argmax grid "
                             "(default: the one it trained with; the paper evaluates on 5)")
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_samples", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, nargs="+", default=[20],
                        help="Euler steps of the flow ODE; one pass per value (CliffordFlow only)")
    parser.add_argument("--seed", type=int, default=0)
    return parser


class PredictTimer:
    """Wraps `model.predict`: fixes the Euler step count and accumulates the time spent inside.

    `functools.wraps` keeps the signature visible to inspect, so the metrics code still finds
    `n_samples` and `cls` on it.
    """

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.original = model.predict
        self.takes_steps = "steps" in inspect.signature(self.original).parameters
        self.steps, self.seconds = None, 0.0

        @functools.wraps(self.original)
        def predict(*args, **kwargs):
            if self.takes_steps and self.steps is not None:
                kwargs["steps"] = self.steps
            self._sync()
            start = time.perf_counter()
            out = self.original(*args, **kwargs)
            self._sync()
            self.seconds += time.perf_counter() - start
            return out

        model.predict = predict

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self, steps):
        self.steps, self.seconds = steps, 0.0


def download_checkpoint(artifact_ref):
    # wandb.Api() reads the artifact without opening a run, so re-running this
    # does not litter the project with empty runs.
    artifact = wandb.Api().artifact(artifact_ref, type="checkpoint")
    directory = Path(artifact.download())
    return next(directory.glob("*.pth"))


def flatten_saved_config(saved):
    """One flat dict from either the nested config (`{"flow": {...}, ...}`) or the old flat one."""
    if any(isinstance(v, dict) for v in saved.values()):
        flat = {}
        for section in saved.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat
    return saved


def _build_i2s_real(saved, eval_rec_level=None):
    from pose3d.models.i2s_real import I2SReal

    # The eval grid is not a weight (non-persistent buffer), so it can be scored on a finer
    # grid than it was trained with: the paper evaluates on rec level 5 (1.875 deg spacing).
    return I2SReal(
        encoder_type=saved.get("encoder", "resnet101"),
        pretrained_backbone=saved.get("pretrained_backbone", True),
        lmax=saved.get("lmax", 6),
        rec_level=saved.get("rec_level", 3),
        eval_rec_level=eval_rec_level or saved.get("i2s_eval_rec_level", 3),
        normalize_input=saved.get("i2s_normalize", True),
    )


def build_model(checkpoint, device, eval_rec_level=None):
    saved = flatten_saved_config(checkpoint.get("config", {}))
    if saved.get("model", saved.get("name")) == "i2s_real":
        model = _build_i2s_real(saved, eval_rec_level)
        result = model.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"Missing keys:    {result.missing_keys[:8]}")
            print(f"Unexpected keys: {result.unexpected_keys[:8]}")
            raise SystemExit("Checkpoint does not match the I2SReal definition.")
        return model.to(device), saved

    algebra = CliffordAlgebra((1, 1, 1))

    model = CliffordFlow(
        algebra,
        # flow_hidden_dim since the no-conv-adapter default; older checkpoints used hidden_dim.
        hidden_dim=saved.get("flow_hidden_dim", saved.get("hidden_dim", [32])),
        n_cond_mv=saved.get("n_cond_mv", 4),
        # The pretrained path wraps the backbone in ImageNetNormalized, which
        # renames its state_dict keys, so this has to match how it was trained.
        pretrained_backbone=saved.get("pretrained_backbone", True),
        encoder_type=saved.get("encoder", "resnet"),
        adapter_grid=saved.get("adapter_grid", saved.get("flow_grid", 16)),
        adapter_channels=saved.get("adapter_channels", 256),
        vector_field_hidden_dim=saved.get("vector_field_hidden_dim"),
        conv_adapter=saved.get("conv_adapter", True),
        mlp_heads=saved.get("mlp_heads", False),
        vector_field=saved.get("vector_field", "clifford"),
        condition_head=saved.get("condition_head", "clifford"),
        cond_tokens=saved.get("cond_tokens", "pooled"),
        so2_channels=saved.get("so2_channels", 128),
        so2_up_token=saved.get("so2_up_token", False),
        so2_split_norm=saved.get("so2_split_norm", False),
        pose_tokens=saved.get("pose_tokens", "rotor"),
        gatr=dict(num_blocks=saved.get("gatr_blocks", 4), mv_channels=saved.get("gatr_mv_channels", 8),
                  s_channels=saved.get("gatr_s_channels", 32), num_heads=saved.get("gatr_heads", 4)),
    )

    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(f"Missing keys:    {result.missing_keys[:8]}")
        print(f"Unexpected keys: {result.unexpected_keys[:8]}")
        raise SystemExit("Checkpoint does not match the model definition -- "
                         "check hidden_dim / n_cond_mv / pretrained_backbone above.")

    return model.to(device), saved


def main():
    args = create_argparser().parse_args()
    torch.manual_seed(args.seed)

    path = Path(args.checkpoint) if args.checkpoint else download_checkpoint(args.artifact)
    print(f"Checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = get_available_device()
    model, saved = build_model(checkpoint, device, eval_rec_level=args.i2s_eval_rec_level)

    print(f"Device: {device}")
    print(f"Trained with: hidden_dim={saved.get('flow_hidden_dim', saved.get('hidden_dim'))}, "
          f"n_cond_mv={saved.get('n_cond_mv')}, "
          f"pretrained_backbone={saved.get('pretrained_backbone')}, "
          f"encoder={saved.get('encoder', 'resnet')}, "
          f"n_epochs={saved.get('n_epochs')}, lr={saved.get('lr')}")

    # The eval helpers only read the device off the config.
    cfg = Config()
    cfg.device = device

    val = Pascal3D(args.path_to_datasets, train=False)
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Test images: {len(val)}")

    timer = PredictTimer(model, device)
    step_counts = args.steps if timer.takes_steps else [None]

    # Warm-up (CUDA context, cuDNN autotuning) so the first timed pass is not charged for it.
    timer.start(step_counts[0])
    calculate_evaluation_metrics(model, [next(iter(val_loader))], cfg, n_samples=max(args.eval_samples))

    names = getattr(getattr(val, "real_dataset", val), "class_names", None)
    rows, per_class = [], {}
    for steps in step_counts:
        for n_samples in args.eval_samples:
            # Same starting noise for every pass, so step counts are compared on equal draws.
            torch.manual_seed(args.seed)
            timer.start(steps)
            err, cls = calculate_evaluation_metrics(model, val_loader, cfg, n_samples=n_samples,
                                                    return_classes=True)
            macro = macro_metrics(err, cls)
            class_mean, medians = macro["class_mean_median_error"], macro["class_medians"]
            per_class[(steps, n_samples)] = medians
            rows.append((steps, n_samples, float(np.median(err)), class_mean, float(np.mean(err)),
                         acc_at(err, 15), acc_at(err, 30),
                         macro["class_mean_acc@15"], macro["class_mean_acc@30"],
                         timer.seconds, 1000 * timer.seconds / len(err)))
            print(f"steps={steps} K={n_samples}: median {rows[-1][2]:.3f}, "
                  f"mean of class medians {class_mean:.3f}, predict {timer.seconds:.1f}s")

    print()
    print("micro = pooled over all test images; macro (cls-*) = averaged over the classes")
    print("predict s = time inside model.predict for the whole test set (data loading excluded)")
    print(f"{'steps':>6} {'samples':>8} {'median':>9} {'cls-mean':>9} {'mean':>9} {'acc@15':>8}"
          f" {'acc@30':>8} {'cls-a@15':>9} {'cls-a@30':>9} {'predict s':>10} {'ms/img':>8}")
    for steps, n_samples, median, class_mean, mean, a15, a30, c15, c30, secs, ms in rows:
        print(f"{str(steps):>6} {n_samples:>8} {median:>9.3f} {class_mean:>9.3f} {mean:>9.3f}"
              f" {a15:>8.3f} {a30:>8.3f} {c15:>9.3f} {c30:>9.3f} {secs:>10.2f} {ms:>8.2f}")

    print()
    print("Per-class median error:")
    keys = list(per_class)
    print(f"{'class':>12}" + "".join(f" {f'{s}/K={k}':>10}" for s, k in keys))
    for c in sorted(per_class[keys[0]]):
        label = names[c] if names else str(c)
        print(f"{label:>12}" + "".join(f" {per_class[key][c]:>10.2f}" for key in keys))

    if args.output_json:
        result = {
            "checkpoint": str(args.checkpoint or args.artifact),
            "device": str(device) if device.type != "cuda" else torch.cuda.get_device_name(device),
            "rows": [dict(zip(("steps", "samples", "median", "class_mean_median", "mean", "acc@15",
                               "acc@30", "class_mean_acc@15", "class_mean_acc@30",
                               "predict_seconds", "predict_ms_per_image"), r))
                     for r in rows],
            "per_class": {f"steps={s},K={k}": {(names[c] if names else str(c)): m
                                               for c, m in v.items()}
                          for (s, k), v in per_class.items()},
        }
        Path(args.output_json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
