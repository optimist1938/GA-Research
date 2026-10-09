"""Training entry point: `python -m pose3d --path_to_datasets=... [flags]`."""

import math
from pathlib import Path

import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.config import Config, parse_args
from pose3d.datasets import create_dataloaders
from pose3d.engine import distributed
from pose3d.engine.checkpoint import form_checkpoint, get_available_device, load_checkpoint
from pose3d.engine.losses import build_criterion
from pose3d.engine.metrics import calculate_evaluation_metrics
from pose3d.engine.tracking import (
    wandb_create_run,
    wandb_finish_run,
    wandb_log_artifact,
    wandb_log_code,
)
from pose3d.engine.trainer import train
from pose3d.models import build_model


def make_algebra(algebra_dim: int = 3) -> CliffordAlgebra:
    algebra_dim = int(algebra_dim)
    if algebra_dim > 6:
        print(
            f"Warning: algebra_dim={algebra_dim} gives mv_dim={2 ** algebra_dim}. "
            "This can significantly increase memory usage and runtime."
        )
    return CliffordAlgebra(tuple([1] * algebra_dim))


def build_scheduler(optimizer, cfg: Config):
    """Linear warmup for `warmup_epochs`, then cosine decay to 5% of the base lr."""
    warmup_epochs = cfg.train.warmup_epochs
    cosine_epochs = cfg.train.n_epochs - warmup_epochs
    if len(optimizer.param_groups) > 1:
        # Several groups (init_mode=backbone): the same shape as below, but as a factor of each
        # group's own lr, so every group warms up from 10% and decays to 5% of its own peak.
        def factor(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, cosine_epochs)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=cfg.effective_lr * 0.05,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def log_model_size(model, run, model_cfg):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    for name in ("condition_head", "vector_field"):
        if hasattr(model, name):
            print(f"  {name}: {sum(p.numel() for p in getattr(model, name).parameters()):,}")

    if run is not None:
        sizes = {"trainable_params": trainable, "world_size": model_cfg.world_size,
                 "per_gpu_batch_size": model_cfg.per_gpu_batch_size,
                 "effective_lr": model_cfg.effective_lr}
        if getattr(model, "adapter", None) is not None:
            sizes["params_excluding_backbone"] = sum(
                p.numel() for name, p in model.named_parameters()
                if not name.startswith("adapter.backbone.")
            )
        run.config.update(sizes, allow_val_change=True)


def init_from_checkpoint(model, cfg: Config):
    """--init_from (or the reflow teacher): start from a trained checkpoint.

    init_mode=strict: the run's flags rebuild the checkpoint's architecture; every tensor is loaded.
    init_mode=backbone: only `adapter.backbone.*` (the fine-tuned ResNet) is loaded and every key of
    it must match; the token head, condition head and vector field keep their fresh initialisation,
    zero read-out included (a different student, e.g. cond_tokens=c4lift + pose_tokens=frame, whose heads would
    receive inputs of a different meaning than the teacher's).
    """
    path = cfg.flow.init_from or cfg.flow.reflow_teacher
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if cfg.flow.init_mode == "strict":
        model.load_state_dict(checkpoint["model"])
    else:
        own = {k for k in model.state_dict() if k.startswith(_BACKBONE)}
        theirs = {k: v for k, v in checkpoint["model"].items() if k.startswith(_BACKBONE)}
        if own != set(theirs):
            raise ValueError(f"backbone keys differ from {path}: missing {sorted(own - set(theirs))[:4]}, "
                             f"unexpected {sorted(set(theirs) - own)[:4]} (check --pretrained_backbone)")
        model.load_state_dict(theirs, strict=False)   # heads keep their fresh init (zero read-out)
        print(f"Initialised the backbone ({len(theirs)} tensors) from {path}; heads start fresh")
    return checkpoint


def attach_reflow_teacher(model, cfg: Config, checkpoint=None):
    """--reflow_teacher: train on the checkpoint's couplings (and start from it, see init_mode)."""
    from pose3d.evaluate import build_model as build_saved_model

    if checkpoint is None:
        checkpoint = torch.load(cfg.flow.reflow_teacher, map_location="cpu", weights_only=False)
    teacher, _ = build_saved_model(checkpoint, cfg.device)
    model.set_reflow_teacher(teacher, cfg.flow.reflow_steps)
    print(f"Reflow: trained on the couplings of {cfg.flow.reflow_teacher} "
          f"({cfg.flow.reflow_steps} teacher steps)")
    return model


_BACKBONE = "adapter.backbone."


def param_groups(model, cfg: Config):
    """One group, or with init_mode=backbone two: the loaded backbone at backbone_lr_mult x the lr,
    everything else (fresh) at the lr. Decided by name, so a resumed run rebuilds the same groups."""
    named = list(model.named_parameters())
    if cfg.flow.init_mode != "backbone" or cfg.flow.backbone_lr_mult == 1.0:
        return [{"params": [p for _, p in named]}]
    return [{"params": [p for n, p in named if n.startswith(_BACKBONE)],
             "lr": cfg.effective_lr * cfg.flow.backbone_lr_mult},
            {"params": [p for n, p in named if not n.startswith(_BACKBONE)]}]


def instantiate(cfg: Config):
    train_loader, val_loader = create_dataloaders(cfg)
    print("Created Tralaloaders")

    algebra = make_algebra(cfg.model.algebra_dim)
    model = build_model(cfg, algebra)

    if cfg.device is None:
        cfg.device = get_available_device()
    model.to(cfg.device)
    checkpoint = None
    if cfg.flow.init_from or cfg.flow.reflow_teacher:
        checkpoint = init_from_checkpoint(model, cfg)
    if cfg.flow.reflow_teacher:
        model = attach_reflow_teacher(
            model, cfg, checkpoint if not cfg.flow.init_from else None)
    model = distributed.convert_sync_bn(model, cfg)

    optimizer = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.effective_lr)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(cfg)

    run = None
    if distributed.is_main():
        print(cfg)
        run = wandb_create_run(cfg)
        print("W&B logging set up completed")
        log_model_size(model, run, cfg)

    return train_loader, val_loader, model, optimizer, scheduler, criterion, run


def main(argv=None):
    cfg = parse_args(argv)
    distributed.maybe_relaunch(cfg, argv)   # with --ddp on several GPUs this becomes torchrun

    torch.backends.cudnn.benchmark = True
    distributed.setup(cfg)
    distributed.seed_everything(cfg)

    train_loader, val_loader, model, optimizer, scheduler, criterion, run = instantiate(cfg)

    path = cfg.run.path_to_checkpoint
    if path is not None:
        try:
            load_checkpoint(model, optimizer, scheduler, path, cfg.device)
            print("Checkpoint successfully loaded. Starting evaluation")
            errors = calculate_evaluation_metrics(model, val_loader, cfg)
            if distributed.is_main():
                torch.save(torch.tensor(errors), "res.pth")
        except Exception as e:
            print(f"Failed to load checkpoint. Starting from scratch. Error: {e}")

    wandb_log_code(run, Path("."))
    torch.cuda.empty_cache()
    # With --ema the averaged copy is the model that was scored, so it is what gets saved.
    model = train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, cfg)

    if cfg.run.save_checkpoint and not cfg.run.sanity_check and distributed.is_main():
        checkpoint_path = form_checkpoint(model, optimizer, scheduler, cfg)
        wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)
    distributed.cleanup()


if __name__ == "__main__":
    main()
