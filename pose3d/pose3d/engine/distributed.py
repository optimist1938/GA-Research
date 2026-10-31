"""Data-parallel training (torch DistributedDataParallel) over every visible GPU.

`--ddp` makes `python -m pose3d ...` relaunch itself under torchrun with one process per GPU;
every helper here is a no-op in a single process, so a 1-GPU / CPU run is unchanged.

The recipe does not depend on the hardware: `train.batch_size` is the GLOBAL batch and each rank
uses `batch_size // world_size`. Only when the global batch is raised on purpose does
`train.lr_scaling` rescale the learning rate.
"""

import os
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pose3d.engine.checkpoint import get_available_device

# A rank stuck in a collective aborts after this instead of burning the Kaggle session.
_TIMEOUT = timedelta(minutes=30)


def launched_by_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def maybe_relaunch(cfg, argv=None):
    """With --ddp on a multi-GPU machine, replace this process by a torchrun launch of itself.

    Returns normally (and does nothing) when --ddp is off, when already under torchrun, or when
    fewer than two GPUs are visible. The RAM cache is built here once, so the ranks only load it.
    """
    if not cfg.features.ddp or launched_by_torchrun():
        return

    n_gpus = torch.cuda.device_count()
    n = min(cfg.distributed.num_gpus or n_gpus, n_gpus)
    if n < 2:
        print(f"--ddp: {n_gpus} GPU(s) visible, training in a single process")
        return

    from pose3d.datasets.pascal import prepare_ram_cache

    argv = list(sys.argv[1:] if argv is None else argv)
    cache_dir = prepare_ram_cache(cfg)
    if cache_dir is not None:
        argv.append(f"--ram_cache_dir={cache_dir}")

    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={n}", "-m", "pose3d", *argv]
    print(f"--ddp: relaunching on {n} GPUs: {' '.join(cmd)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, cmd)


def setup(cfg):
    """Join the process group when launched by torchrun and pick this process's device.

    Sets cfg.device, cfg.rank and cfg.world_size.
    """
    if not launched_by_torchrun():
        cfg.device = get_available_device()
        return

    if not cfg.features.ddp:
        raise RuntimeError("Launched under torchrun without --ddp; every rank would train the "
                           "same model on its own. Pass --ddp.")

    if not cfg.distributed.nccl_p2p:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        cfg.device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        cfg.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, timeout=_TIMEOUT)

    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    if cfg.train.batch_size % cfg.world_size:
        raise ValueError(f"--batch_size {cfg.train.batch_size} (global) is not divisible by "
                         f"{cfg.world_size} GPUs")
    if is_main():
        print(f"DDP: {cfg.world_size} processes, per-GPU batch {cfg.per_gpu_batch_size}, "
              f"effective lr {cfg.effective_lr:g}")


def seed_everything(cfg):
    """Seed torch/numpy/random with run.seed + rank (a no-op without --seed)."""
    seed = cfg.run.seed
    if seed is None:
        return
    import random
    seed += cfg.rank
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def wrap_ddp(module, cfg):
    """DDP around `module` when there is more than one process, else `module` itself."""
    if cfg.world_size == 1:
        return module
    device_ids = [cfg.device.index] if cfg.device.type == "cuda" else None
    return DistributedDataParallel(
        module, device_ids=device_ids,
        find_unused_parameters=cfg.distributed.ddp_find_unused)


def convert_sync_bn(model, cfg):
    if cfg.world_size > 1 and cfg.distributed.sync_bn:
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def all_reduce_sum(values, cfg):
    """Sum a list of floats over all ranks."""
    if not is_distributed():
        return values
    t = torch.tensor(values, dtype=torch.float64, device=cfg.device)
    dist.all_reduce(t)
    return t.tolist()


def sync_buffers(model):
    """Copy rank 0's buffers (BatchNorm running statistics) to every rank before evaluation."""
    if not is_distributed():
        return
    gpu = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else None
    for buf in model.buffers():
        if gpu is not None and buf.device.type == "cpu":
            # NCCL broadcasts CUDA tensors only; AveragedModel keeps its n_averaged counter on the CPU.
            moved = buf.to(gpu)
            dist.broadcast(moved, src=0)
            buf.copy_(moved.cpu())
        else:
            dist.broadcast(buf, src=0)


def gather_errors(err, cfg=None):
    """Concatenate the per-rank error arrays (order is irrelevant for the median / Acc@theta)."""
    if not is_distributed():
        return err
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, np.asarray(err))
    return np.hstack(parts)


def cleanup():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()
