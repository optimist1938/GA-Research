"""Training / validation loop."""

import inspect
import time

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from pose3d.engine.distributed import (
    all_reduce_sum, is_main, sync_buffers, wrap_ddp,
)
from pose3d.datasets.modelnet_points import CLASSES as MODELNET10_CLASSES
from pose3d.datasets.symsol import SHAPE_SETS
from pose3d.engine.likelihood import LOG_UNIFORM, symsol_log_likelihood
from pose3d.engine.metrics import acc_at, calculate_evaluation_metrics, macro_metrics
from pose3d.engine.tracking import log_offline_sync


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _supports_class_argument(method) -> bool:
    params = list(inspect.signature(method).parameters.values())
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 2


def _compute_loss(model, data, criterion, cfg):
    img = data["img"].to(cfg.device)
    targets = data["rot"].to(cfg.device)

    clas = None
    if "cls" in data:
        clas = data["cls"].to(cfg.device)

    if hasattr(model, "compute_loss") and callable(getattr(model, "compute_loss")):
        # Important: avoid a duplicate forward pass before model.compute_loss(),
        # which can skew BatchNorm running statistics during training.
        params = inspect.signature(model.compute_loss).parameters
        extra = {}
        if getattr(cfg.data, "symsol_nearest_coupling", False) and "rots_eq" in data and "rots_eq" in params:
            extra = dict(rots_eq=data["rots_eq"].to(cfg.device), n_eq=data["n_eq"].to(cfg.device))
        if clas is not None and "cls" in params:
            return model.compute_loss(img, targets, criterion, cls=clas, **extra)
        return model.compute_loss(img, targets, criterion, **extra)

    if clas is not None and _supports_class_argument(model.forward):
        outputs = model(img, clas)
    else:
        outputs = model(img)

    if cfg.train.loss == "prob":
        idx = model.get_nearest_idx(targets).long().view(-1)
        return criterion(outputs, idx)
    return criterion(outputs, targets)


class TrainStep(torch.nn.Module):
    """Model + criterion as one module whose forward is the loss.

    The trainer calls `model.compute_loss`, not `model.forward`, and DDP only synchronises
    gradients for what goes through its own forward, so DDP wraps this instead of the model.
    """

    def __init__(self, model, criterion, cfg):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.cfg = cfg

    def forward(self, data):
        return _compute_loss(self.model, data, self.criterion, self.cfg)


def build_step(model, criterion, cfg):
    return wrap_ddp(TrainStep(model, criterion, cfg), cfg)


def train_epoch(model, loader, optimizer, criterion, cfg, step=None, ema=None):
    total_loss = 0.0
    n_objects = 0
    device_type = "cuda" if cfg.device.type == "cuda" else "cpu"

    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None

    if step is None:
        step = build_step(model, criterion, cfg)
    step.train()
    for data in tqdm(loader, disable=not is_main() or cfg.run.platform == "kaggle"):
        optimizer.zero_grad(set_to_none=True)

        loss = step(data)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update_parameters(model)

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    total_loss, n_objects = all_reduce_sum([total_loss, n_objects], cfg)
    return total_loss / max(n_objects, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, cfg):
    total_loss = 0.0
    n_objects = 0

    model.eval()
    for data in tqdm(loader, disable=not is_main() or cfg.run.platform == "kaggle"):
        loss = _compute_loss(model, data, criterion, cfg)

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    total_loss, n_objects = all_reduce_sum([total_loss, n_objects], cfg)
    return total_loss / max(n_objects, 1)


def log_viz(model, val_loader, run, cfg, final=False, commit=False):
    """Flow-trajectory visualizations to W&B (see engine/flow_viz.py). A failure here is reported and never stops the run."""
    if run is None or not cfg.run.log_viz or not is_main() or not hasattr(model, "velocity"):
        return
    try:
        from pose3d.engine.flow_viz import wandb_media
        run.log(wandb_media(model, val_loader.dataset, n=cfg.run.viz_images, final=final), commit=commit)
    except Exception as e:
        print(f"Visualization logging skipped: {type(e).__name__}: {e}")


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, cfg):
    """Train, then score; returns the model that was evaluated (the EMA copy with --ema)."""
    n_epochs = cfg.train.n_epochs
    step = build_step(model, criterion, cfg)
    # BatchNorm running statistics are averaged too (use_buffers), so the EMA copy is
    # evaluated with statistics that match its weights.
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.train.ema_decay), use_buffers=True)
           if cfg.features.ema else None)
    eval_model = ema.module if ema is not None else model
    for i in range(n_epochs):
        t0 = time.time()
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(i)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg, step=step, ema=ema)
        sync_buffers(model)   # rank 0's BatchNorm statistics, so every shard is scored alike
        if ema is not None:
            sync_buffers(ema)
        val_loss = validate_epoch(eval_model, val_loader, criterion, cfg)
        err, cls = calculate_evaluation_metrics(eval_model, val_loader, cfg, return_classes=True)
        mre = np.median(err).__float__()
        macro = macro_metrics(err, cls) if cls is not None else None

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "median_rotation_error": mre,
            "acc@15": acc_at(err, 15),   # the share of validation images within 15 deg (primary metric on SYMSOL)
            "acc@30": acc_at(err, 30),
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm(model),
            "epoch_time": time.time() - t0,
        }
        if macro is not None:
            metrics["class_mean_median_error"] = macro["class_mean_median_error"]
        if cfg.run.viz_every > 0 and (i + 1) % cfg.run.viz_every == 0:
            log_viz(model, val_loader, run, cfg, commit=False)   # joins this epoch's step instead of opening one
        if run is not None:
            run.log(metrics)
        scheduler.step()
        if is_main():
            print(
                f"Training on {cfg.device} epoch {i + 1} / {n_epochs}. \n"
                f"Train loss {train_loss}, val loss {val_loss}\n"
                f"Median rotation error {mre}, acc@15 {metrics['acc@15']:.4f}, acc@30 {metrics['acc@30']:.4f} "
                f"({time.time() - t0:.0f}s)"
            )
            if macro is not None:
                print(f"Class-mean median rotation error {macro['class_mean_median_error']}")
            log_offline_sync(run, cfg, step=i, metrics=metrics)

    final_evaluation(eval_model, val_loader, run, cfg, raw_model=model if ema is not None else None)
    log_viz(eval_model, val_loader, run, cfg, final=True, commit=True)
    if cfg.run.dataset == "symsol" and hasattr(eval_model, "grid_logits") and is_main():
        from pose3d.engine.symsol_ll import symsol_final_log_likelihood
        symsol_final_log_likelihood(eval_model, cfg, run)
    return eval_model


PASCAL3D_CLASSES = ('aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
                    'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor')


def final_evaluation(model, val_loader, run, cfg, raw_model=None):
    '''Re-score the trained model with multi-sample prediction.

    Per-epoch metrics use a single draw so they stay cheap; a generative model
    deserves a proper mode estimate once, at the end.
    '''
    n_samples = cfg.eval_samples
    if n_samples <= 1:
        return None

    err, cls = calculate_evaluation_metrics(model, val_loader, cfg, n_samples=n_samples,
                                            return_classes=True)
    metrics = {
        "final_median_rotation_error": float(np.median(err)),
        "final_acc@15": acc_at(err, 15),
        "final_acc@30": acc_at(err, 30),
        "final_eval_samples": n_samples,
    }
    macro = macro_metrics(err, cls) if cls is not None else None
    if macro is not None:
        # Macro (class-averaged) counterparts of the pooled numbers above; the median is the
        # published Pascal3D+ metric (mean of per-class medians), see per_class_median.
        metrics["final_class_mean_median_error"] = macro["class_mean_median_error"]
        metrics["final_class_mean_acc@15"] = macro["class_mean_acc@15"]
        metrics["final_class_mean_acc@30"] = macro["class_mean_acc@30"]
        for c, m in macro["class_medians"].items():
            metrics[f"final_median_error_class{c}"] = m
            metrics[f"final_acc@15_class{c}"] = macro["class_acc@15"][c]
            metrics[f"final_acc@30_class{c}"] = macro["class_acc@30"][c]
    # With --ema, the last iterate too, so one run shows what the averaging bought.
    if raw_model is not None:
        err_raw, cls_raw = calculate_evaluation_metrics(raw_model, val_loader, cfg, n_samples=n_samples,
                                                        return_classes=True)
        metrics["final_median_rotation_error_raw"] = float(np.median(err_raw))
        macro_raw = macro_metrics(err_raw, cls_raw) if cls_raw is not None else None
        if macro_raw is not None:
            metrics["final_class_mean_median_error_raw"] = macro_raw["class_mean_median_error"]

    if cfg.run.dataset in ("symsol", "symsol_points") and cfg.data.symsol_likelihood:
        # The SYMSOL metric (IPDF / Image2Sphere / Howell et al.): mean log-likelihood over each image's
        # equivalent poses, averaged per shape and then over shapes.
        d = cfg.data
        if hasattr(model, "grid_logits"):
            # A distribution over Image2Sphere's grid (I2SReal): read off the histogram, as upstream does.
            from pose3d.engine.grid_model_likelihood import grid_model_log_likelihood
            ll, ll_cls = grid_model_log_likelihood(model, val_loader, cfg.device, n_equivalents=d.symsol_ll_equivalents)
        else:
            try:
                ll, ll_cls = symsol_log_likelihood(model, val_loader, cfg.device, n_equivalents=d.symsol_ll_equivalents,
                                                   steps=d.symsol_ll_steps)
            except Exception:   # the rotation errors above and the checkpoint must survive a likelihood failure
                import traceback
                traceback.print_exc()
                print("SYMSOL log-likelihood failed (see the traceback above); final_log_likelihood is NaN")
                ll, ll_cls = np.full(1, np.nan), np.zeros(1, dtype=int)
        per_shape = {int(c): float(ll[ll_cls == c].mean()) for c in np.unique(ll_cls)}
        metrics["final_log_likelihood"] = float(np.mean(list(per_shape.values())))
        for c, v in per_shape.items():
            metrics[f"final_log_likelihood_class{c}"] = v
        if is_main():
            names = SHAPE_SETS[d.symsol_set]
            print(f"SYMSOL log-likelihood (mean over shapes, uniform = {LOG_UNIFORM:.2f}): "
                  f"{metrics['final_log_likelihood']:.3f}  "
                  + "  ".join(f"{names[c]} {v:.3f}" for c, v in per_shape.items()))

    if is_main():
        print(
            f"Final evaluation with {n_samples} samples per image: "
            f"median rotation error {metrics['final_median_rotation_error']}, "
            f"acc@15 {metrics['final_acc@15']}, acc@30 {metrics['final_acc@30']}"
        )
        if macro is not None:
            print(
                f"Class-averaged (macro): mean of per-class medians "
                f"{macro['class_mean_median_error']}, acc@15 {macro['class_mean_acc@15']}, "
                f"acc@30 {macro['class_mean_acc@30']}"
            )
            if "final_median_rotation_error_raw" in metrics:
                print(f"Last iterate (no EMA): median {metrics['final_median_rotation_error_raw']}, "
                      f"class mean {metrics.get('final_class_mean_median_error_raw')}")
            print(f"{'class':>12} {'median':>8} {'acc@15':>8} {'acc@30':>8}")
            for c, m in macro["class_medians"].items():
                name = (PASCAL3D_CLASSES[c] if cfg.run.dataset == "pascal" and c < len(PASCAL3D_CLASSES)
                        else SHAPE_SETS[cfg.data.symsol_set][c] if cfg.run.dataset in ("symsol", "symsol_points")
                        else MODELNET10_CLASSES[c] if cfg.run.dataset.startswith("modelnet10") else str(c))
                print(f"{name:>12} {m:>8.2f} {macro['class_acc@15'][c]:>8.3f} {macro['class_acc@30'][c]:>8.3f}")
        log_offline_sync(run, cfg, step=cfg.train.n_epochs, metrics=metrics, final=True)
    if run is not None:
        run.summary.update(metrics)
    return metrics
