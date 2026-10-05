"""Windowed SGD training for the differentiable PA models.

Both model families are pointwise in ``k`` apart from a handful of sample
delays and the final band-limiting FIR, so training on random contiguous
windows is exact up to edge effects.  ``halo`` trims the affected samples from
the loss, which keeps the objective identical to the full-signal one.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from .matlab import nmse_db

__all__ = ["TrainConfig", "nmse_loss", "train", "evaluate", "Logger", "WandbLogger", "resolve_device"]


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA when it is there, otherwise CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class Logger:
    """Minimal sink for training metrics; the default prints and keeps history.

    Kept as a plain object rather than a framework dependency so the package
    runs with nothing but numpy/scipy/torch installed.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.history: list[dict] = []

    def log(self, row: dict) -> None:
        self.history.append(row)
        if self.verbose:
            print(
                f"  step {row['step']:5d}  batch {row['loss_db']:8.3f} dB"
                f"  train {row['train_db']:8.3f} dB  val {row['val_db']:8.3f} dB"
                f"  ({row['elapsed_s']:.0f}s)",
                flush=True,
            )

    def finish(self, summary: dict) -> None:
        pass


class WandbLogger(Logger):
    """Optional Weights & Biases sink.

    ``wandb`` is imported lazily, so it is only needed when this class is used.
    The sibling pose-estimation project in this repo logs to the ``clifforders``
    entity; the same defaults are used here for continuity.
    """

    def __init__(self, project: str = "dpd", entity: str | None = "clifforders", config: dict | None = None, **kwargs):
        super().__init__(verbose=True)
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, entity=entity, config=config or {}, **kwargs)

    def log(self, row: dict) -> None:
        super().log(row)
        self._wandb.log({k: v for k, v in row.items() if k != "step"}, step=row["step"])

    def finish(self, summary: dict) -> None:
        self._run.summary.update(summary)
        self._wandb.finish()


@dataclasses.dataclass
class TrainConfig:
    steps: int = 2000
    window: int = 8192
    lr: float = 3e-3
    weight_decay: float = 0.0
    halo: int = 32
    log_every: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    scheduler: str = "cosine"


def nmse_loss(d: torch.Tensor, y: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """The objective the reference NMSE reports, as a differentiable loss."""
    return (d - y).abs().pow(2).sum() / x_ref.abs().pow(2).sum()


@torch.no_grad()
def evaluate(model, x: torch.Tensor, d: torch.Tensor, x_ref: np.ndarray, index: slice, chunk: int = 32768) -> float:  # noqa: D401
    """Full-signal NMSE over ``index``, evaluated in chunks with a halo."""
    model.eval()
    y = torch.zeros_like(d)
    n = x.shape[1]
    halo = 64
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        lo, hi = max(0, a - halo), min(n, b + halo)
        out = model(x[:, lo:hi])
        y[a:b] = out[a - lo : a - lo + (b - a)]
    model.train()
    err = (d - y).cpu().numpy()[index]
    return nmse_db(np.asarray(x_ref)[index], err)


def train(
    model,
    x: torch.Tensor,
    d: torch.Tensor,
    x_ref: np.ndarray,
    split,
    cfg: TrainConfig,
    verbose: bool = True,
    logger: "Logger | None" = None,
):
    """Train ``model`` on ``split.train`` and report NMSE on both halves.

    Returns the history list; the model is left holding its final parameters.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
        if cfg.scheduler == "cosine"
        else None
    )

    lo, hi, _ = split.train.indices(x.shape[1])
    if hi - lo <= cfg.window + 2 * cfg.halo:
        raise ValueError(f"training region ({hi - lo} samples) is too short for window={cfg.window}")

    x_ref_t = torch.as_tensor(np.asarray(x_ref), dtype=x.dtype, device=x.device)
    logger = logger if logger is not None else Logger(verbose=verbose)
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        a = int(rng.integers(lo, hi - cfg.window))
        b = a + cfg.window
        xw, dw = x[:, a:b], d[a:b]
        y = model(xw)
        h = cfg.halo
        # The NMSE reference is the *unnormalised* xRef, exactly as the MATLAB
        # script captures it before scaling the rows to unit peak.  Using the
        # normalised row here would only rescale the loss by a constant, but it
        # would make the logged batch figure incomparable with the reported NMSE.
        loss = nmse_loss(dw[h:-h], y[h:-h], x_ref_t[a + h : b - h])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if sched is not None:
            sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            logger.log(
                {
                    "step": step,
                    "loss_db": 10 * np.log10(loss.item()),
                    "train_db": evaluate(model, x, d, x_ref, split.train),
                    "val_db": evaluate(model, x, d, x_ref, split.val),
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_s": time.time() - t0,
                }
            )
    return logger.history
