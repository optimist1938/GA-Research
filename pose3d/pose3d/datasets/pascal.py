"""Pascal3D+ data loading (image2sphere's Pascal3D) with the RAM-cache and augmentation options."""

import gc
import pathlib
import tempfile
import time

import numpy as np
import torch
from image2sphere.pascal_dataset import Pascal3D
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from pose3d.datasets.cache import InMemoryDataset, RawImageCache


class PascalSanityCheckDataset(Dataset):
    """The first `batch_size` training samples; used by --sanity_check."""

    def __init__(self, cfg):
        self.base_dataset = Pascal3D(datasets_dir=cfg.run.path_to_datasets, train="train")
        self.size = cfg.train.batch_size

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i < self.size:
            return self.base_dataset[i]
        raise ValueError("List Index out of Range")


def _cache_file(directory, split):
    return pathlib.Path(directory) / f"pascal_{split}.pt" if directory else None


def _num_builder(cfg):
    return 4 if cfg.run.platform == "kaggle" else 2


def _load_cached(split, cfg):
    """The cached tensors for `split` if --ram_cache_dir has them (Pascal3D is not even
    constructed, which skips parsing every annotation .mat), else None."""
    cached = _cache_file(cfg.data.ram_cache_dir, split)
    if cached is None or not cached.exists():
        return None
    t0 = time.time()
    ds = InMemoryDataset.load(cached, include_cls=cfg.features.fisher_prior)
    print(f"[timing] {split}: loaded {len(ds)} samples from cache {cached} in {time.time() - t0:.1f}s")
    return ds


def _save_cache(ds, split, cfg):
    target = _cache_file(cfg.data.ram_cache_save_dir, split)
    if target is not None:
        t0 = time.time()
        ds.save(target)
        print(f"[timing] {split}: saved cache to {target} in {time.time() - t0:.1f}s")


def _bound_synthetic_pool(train, max_synth):
    if len(train.synth_dataset.files) == 0:
        raise FileNotFoundError(
            "--use_synth found no RenderForCNN images under "
            "<path_to_datasets>/syn_images_cropped_bkg_overlaid/<synset>/*/*.png"
        )
    available = len(train.synth_dataset.files)
    if 0 < max_synth < available:
        # Bound the pool before the cache sees it. Each epoch only draws
        # 3 * len(real) synthetic samples, so a capped pool still gives
        # every draw a fresh image.
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(available, size=max_synth, replace=False))
        train.synth_dataset.files = [train.synth_dataset.files[i] for i in keep]
    print(f"Synthetic pool: {len(train.synth_dataset.files)} of {available} renders, "
          f"{3 * len(train.real_dataset)} drawn per epoch")


def _train_dataset(cfg):
    f, d = cfg.features, cfg.data
    if d.synth_pack_dir and (f.use_warp or f.use_synth):
        # Image2Sphere's Pascal3D(train=True, use_warp, use_synth), read from RAM and augmented on
        # every access (the RAM cache below would freeze one draw of the augmentation).
        from pose3d.datasets.packed import PackedPascal3D
        t0 = time.time()
        ds = PackedPascal3D(d.synth_pack_dir, use_warp=f.use_warp, use_synth=f.use_synth,
                            use_weights=d.synth_pack_weights, max_synth=d.max_synth)
        n_syn = len(ds.synth_dataset) if ds.synth_dataset is not None else 0
        print(f"[timing] train: synth pack, {len(ds.real_dataset)} real + {n_syn} synthetic images, "
              f"{len(ds)} samples per epoch, loaded in {time.time() - t0:.1f}s")
        return ds
    if f.ram_memory and not f.raw_cache:
        cached = _load_cached("train", cfg)
        if cached is not None:
            if f.use_warp or f.use_synth:
                print("WARNING: the cached train tensors freeze one draw of the augmentation "
                      "(the cache key does not encode use_warp/use_synth).")
            return cached

    train = Pascal3D(cfg.run.path_to_datasets, train=True,
                     use_warp=f.use_warp, use_synth=f.use_synth)
    if f.use_synth:
        _bound_synthetic_pool(train, d.max_synth)

    if not f.ram_memory:
        return train

    num_builder = _num_builder(cfg)
    if f.raw_cache:
        # Cache the file reads and let Pascal3D augment on every access, so
        # augmentation is unlimited rather than a fixed pool of draws.
        cache = RawImageCache.for_dataset(train, workers=2 * num_builder).install()
        print(f"Raw cache: {cache.nbytes / 2**30:.2f} GiB held in RAM, augmentation runs per access")
        return train

    if f.use_warp and d.cache_draws == 1:
        print("WARNING: --ram_memory caches one draw per image, which freezes the "
              "augmentation --use_warp just enabled. Pass --cache_draws > 1 or --raw_cache.")
    t0 = time.time()
    ds = InMemoryDataset(train, build_workers=num_builder,
                         use_multiprocessing=d.multiprocessing,
                         n_draws=d.cache_draws, include_cls=f.fisher_prior)
    print(f"[timing] train: built {len(ds)} samples from Pascal3D in {time.time() - t0:.1f}s")
    _save_cache(ds, "train", cfg)
    return ds


def _class_labels(real):
    """Class index of every sample of a Pascal3DReal, from the annotation files alone."""
    names = real.class_names
    return [names.index(loadmat(a)["record"]["objects"][0][0][0][0]["class"][0]) for a in real.annot_paths]


def _classes_match_cache(ds, real, classes, atol=1e-3):
    """Spot-check that `classes` line up with a cache's samples: the first and last sample of
    every class must have the cached ground-truth rotation of the same index in `real`."""
    classes = np.asarray(classes)
    picks = sorted({int(i) for c in np.unique(classes)
                    for i in (np.flatnonzero(classes == c)[0], np.flatnonzero(classes == c)[-1])})
    return all(torch.allclose(real[i]["rot"].float(), ds.targets[i].float(), atol=atol) for i in picks)


def _attach_eval_classes(ds, cfg, real=None):
    """Give the validation set per-sample class labels (batch["cls_eval"]) for macro metrics.

    The pre-built RAM caches hold no labels, and rebuilding one is not always possible, so they
    are read from the annotations of the mounted Pascal3D+ (no image is decoded), and the
    ground-truth rotations of a few samples per class are compared with the cache to make sure
    the two orders agree. Without the dataset, or on any mismatch, macro metrics are skipped.
    """
    if getattr(ds, "eval_clss", None) is not None or getattr(ds, "clss", None) is not None:
        return
    verify = real is None
    try:
        if real is None:
            root = pathlib.Path(cfg.run.path_to_datasets) / "PASCAL3D+_release1.1"
            if not root.is_dir():
                print("Per-class metrics skipped: Pascal3D+ is not mounted, the cache has no labels")
                return
            real = Pascal3D(cfg.run.path_to_datasets, train=False).real_dataset
        classes = _class_labels(real)
        if len(classes) != len(ds):
            print(f"Per-class metrics skipped: {len(classes)} annotations for {len(ds)} cached samples")
            return
        if verify and not _classes_match_cache(ds, real, classes):
            print("Per-class metrics skipped: the annotations do not line up with the cache")
            return
        ds.set_eval_classes(classes)
    except Exception as e:   # metrics are a report; they must never stop a run
        print(f"Per-class metrics skipped: {type(e).__name__}: {e}")


def _val_dataset(cfg):
    f = cfg.features
    if f.ram_memory:
        cached = _load_cached("val", cfg)
        if cached is not None:
            _attach_eval_classes(cached, cfg)
            return cached

    # Pascal3D asserts use_warp/use_synth are off for the test split.
    val = Pascal3D(cfg.run.path_to_datasets, train=False)
    if not f.ram_memory:
        return val

    # Validation stays deterministic: one draw, no augmentation.
    t0 = time.time()
    ds = InMemoryDataset(val, build_workers=_num_builder(cfg), include_cls=f.fisher_prior)
    print(f"[timing] val: built {len(ds)} samples from Pascal3D in {time.time() - t0:.1f}s")
    _save_cache(ds, "val", cfg)
    _attach_eval_classes(ds, cfg, real=val.real_dataset)
    return ds


def prepare_ram_cache(cfg):
    """Build the RAM-cache tensors once, in the process that launches the DDP ranks.

    Returns the directory holding `pascal_{train,val}.pt` for the ranks to load (their
    --ram_cache_dir), or None when they should build their own (no RAM cache, the raw cache,
    a sanity check or another dataset).
    """
    f, d = cfg.features, cfg.data
    if cfg.run.dataset != "pascal" or cfg.run.sanity_check or not f.ram_memory or f.raw_cache:
        return None
    if all(_cache_file(d.ram_cache_dir, s) is not None and _cache_file(d.ram_cache_dir, s).exists()
           for s in ("train", "val")):
        return d.ram_cache_dir

    cache_dir = tempfile.mkdtemp(prefix="pose3d_ram_cache_")
    d.ram_cache_dir, d.ram_cache_save_dir = None, cache_dir   # rebuild both, save both
    print(f"Building the RAM cache once for all ranks in {cache_dir}")
    train, val = _train_dataset(cfg), _val_dataset(cfg)
    del train, val
    gc.collect()
    return cache_dir


def _num_workers(cfg):
    if cfg.data.num_workers is not None:
        return cfg.data.num_workers
    f = cfg.features
    # The raw cache moves the warp into the workers, so it wants the full count.
    base = 2 if (f.ram_memory and not f.raw_cache) else 4
    return max(1, base // cfg.world_size)


def create_dataloaders(cfg):
    if cfg.run.dataset == "dummynet":
        from pose3d.datasets.modelnet import DummyPointCloudDataset
        train_dataset = DummyPointCloudDataset(cfg, size=1000)
        val_dataset = DummyPointCloudDataset(cfg, size=100)
    elif cfg.run.dataset in ("modelnet10", "symsol"):
        from pose3d.datasets.benchmarks import create_benchmark_datasets
        train_dataset, val_dataset = create_benchmark_datasets(cfg)
    elif not cfg.run.sanity_check:
        train_dataset = _train_dataset(cfg)
        val_dataset = _val_dataset(cfg)
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(cfg)

    num_workers = _num_workers(cfg)
    persistent_workers = num_workers > 0
    batch_size = cfg.per_gpu_batch_size   # train.batch_size is the global batch

    if cfg.world_size > 1:
        # Every rank gets the same number of samples (drop_last), so DDP never waits on a rank
        # that ran out of batches; the shuffle is reseeded per epoch through set_epoch. The
        # validation set is split into disjoint strided shards and re-joined by the metrics.
        sampler = DistributedSampler(train_dataset, num_replicas=cfg.world_size, rank=cfg.rank,
                                     shuffle=True, seed=cfg.run.seed or 0, drop_last=True)
        val_dataset = Subset(val_dataset, range(cfg.rank, len(val_dataset), cfg.world_size))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, sampler=sampler,
                                  persistent_workers=persistent_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, shuffle=True,
                                  persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=False, persistent_workers=persistent_workers)
    return train_loader, val_loader
