"""Re-score a saved CliffordFlow checkpoint with multi-sample prediction.

    python -m pose3d.evaluate --artifact <entity/project/name.pth:vN> --path_to_datasets ...

Downloads the checkpoint from W&B and runs one evaluation pass over the Pascal3D test split
at several sample counts, so the gain from mode selection (the geodesic medoid) is
measurable on the same weights. Nothing is trained and no W&B run is created.

Checkpoints written before the nested config store a flat `config` dict; both layouts load.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader

from pose3d.config import Config
from pose3d.engine.checkpoint import get_available_device
from pose3d.engine.metrics import acc_at, calculate_evaluation_metrics
from pose3d.models.clifford_flow import CliffordFlow


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=str,
                        default="clifforders/3D Pose Estimation/clifford_flow_pretrained.pth:v1")
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_samples", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--seed", type=int, default=0)
    return parser


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


def build_model(checkpoint, device):
    saved = flatten_saved_config(checkpoint.get("config", {}))
    algebra = CliffordAlgebra((1, 1, 1))

    model = CliffordFlow(
        algebra,
        hidden_dim=saved.get("hidden_dim", [32]),
        n_cond_mv=saved.get("n_cond_mv", 4),
        # The pretrained path wraps the backbone in ImageNetNormalized, which
        # renames its state_dict keys, so this has to match how it was trained.
        pretrained_backbone=saved.get("pretrained_backbone", True),
        encoder_type=saved.get("encoder", "resnet"),
        adapter_grid=saved.get("adapter_grid", saved.get("flow_grid", 16)),
        adapter_channels=saved.get("adapter_channels", 256),
        vector_field_hidden_dim=saved.get("vector_field_hidden_dim"),
        condition_head=saved.get("condition_head", True),
        mlp_heads=saved.get("mlp_heads", False),
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

    path = download_checkpoint(args.artifact)
    print(f"Checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = get_available_device()
    model, saved = build_model(checkpoint, device)

    print(f"Device: {device}")
    print(f"Trained with: hidden_dim={saved.get('hidden_dim')}, "
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

    rows = []
    for n_samples in args.eval_samples:
        err = calculate_evaluation_metrics(model, val_loader, cfg, n_samples=n_samples)
        rows.append((n_samples, float(np.median(err)), float(np.mean(err)),
                     acc_at(err, 15), acc_at(err, 30)))
        print(f"K={n_samples}: median {rows[-1][1]:.3f}")

    print()
    print(f"{'samples':>8} {'median':>9} {'mean':>9} {'acc@15':>8} {'acc@30':>8}")
    for n_samples, median, mean, a15, a30 in rows:
        print(f"{n_samples:>8} {median:>9.3f} {mean:>9.3f} {a15:>8.3f} {a30:>8.3f}")


if __name__ == "__main__":
    main()
