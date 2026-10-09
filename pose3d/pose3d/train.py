"""Training entry point: `python -m pose3d --path_to_datasets=... [flags]`."""

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


def attach_reflow_teacher(model, cfg: Config):
    """--reflow_teacher: start from the checkpoint's weights and train on its couplings."""
    from pose3d.evaluate import build_model as build_saved_model

    checkpoint = torch.load(cfg.flow.reflow_teacher, map_location="cpu", weights_only=False)
    teacher, _ = build_saved_model(checkpoint, cfg.device)
    if cfg.flow.reflow_init == "strict":
        # The run's flags have to rebuild the teacher's architecture.
        model.load_state_dict(checkpoint["model"])
        model._fresh_params = set()
    else:
        # A different student (e.g. cond_tokens=c4lift, pose_tokens=frame): copy every tensor whose
        # name and shape match, leave the rest at their initialisation.
        own = model.state_dict()
        shared = {k: v for k, v in checkpoint["model"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(shared, strict=False)
        model._fresh_params = {n for n, _ in model.named_parameters() if n not in shared}
        print(f"Reflow partial init: {len(shared)} / {len(own)} tensors from the teacher; "
              f"fresh parameters: {sorted(model._fresh_params)}")
    model.set_reflow_teacher(teacher, cfg.flow.reflow_steps)
    print(f"Reflow: initialised from and trained on the couplings of {cfg.flow.reflow_teacher} "
          f"({cfg.flow.reflow_steps} teacher steps)")
    return model


def param_groups(model, cfg: Config):
    """One group, or with a partial reflow init two: the parameters the teacher did not provide
    train at fresh_lr_mult x the learning rate (they start from scratch while the rest fine-tune)."""
    fresh = getattr(model, "_fresh_params", set())
    mult = cfg.flow.fresh_lr_mult
    if not fresh or mult == 1.0:
        return [{"params": [p for p in model.parameters()]}]
    named = list(model.named_parameters())
    return [{"params": [p for n, p in named if n not in fresh]},
            {"params": [p for n, p in named if n in fresh], "lr": cfg.effective_lr * mult}]


def instantiate(cfg: Config):
    train_loader, val_loader = create_dataloaders(cfg)
    print("Created Tralaloaders")

    algebra = make_algebra(cfg.model.algebra_dim)
    model = build_model(cfg, algebra)

    if cfg.device is None:
        cfg.device = get_available_device()
    model.to(cfg.device)
    if cfg.flow.reflow_teacher:
        model = attach_reflow_teacher(model, cfg)
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
