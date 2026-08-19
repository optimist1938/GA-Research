from image2sphere.pascal_dataset import Pascal3D
import image2sphere.pascal_dataset as pascal_dataset_module
import torch
from tqdm import tqdm
import io
import multiprocessing as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pathlib
from src.img_to_pcd_stuff import MeshProcessor
from src.evaluation_metrics import project_to_orthogonal_manifold, create_technical_matrices
import pandas as pd


class PascalSanityCheckDataset(Dataset):
    def __init__(self, config):
        self.base_dataset = Pascal3D(datasets_dir=config.path_to_datasets, train="train")
        self.size = config.batch_size


    def __len__(self):
        return self.size


    def __getitem__(self, i):
        if i < self.size:
            return self.base_dataset[i]
        raise ValueError("List Index out of Range")
    

def _collate_keep(img_key="img", rot_key="rot"):
    def _c(batch):
        imgs = torch.stack([b[img_key] for b in batch], dim=0)
        rots = torch.stack([b[rot_key] for b in batch], dim=0)
        return imgs, rots
    return _c


def _iter_chunks(n, chunk_size):
    for start in range(0, n, chunk_size):
        yield list(range(start, min(start + chunk_size, n)))


def _load_chunk(args):
    base, indices, img_key, rot_key = args
    imgs = []
    rots = []
    for idx in indices:
        sample = base[idx]
        imgs.append(sample[img_key])
        rots.append(sample[rot_key])
    return torch.stack(imgs, dim=0), torch.stack(rots, dim=0)


class _CachedPIL:
    '''Stands in for the PIL object Pascal3DReal.__getitem__ consumes.'''

    def __init__(self, arr):
        self._arr = arr
        self.size = (arr.shape[1], arr.shape[0])   # PIL reports (width, height)

    def convert(self, mode):
        # Upstream discards the return value, so matching that is enough.
        return self

    def getdata(self):
        # Upstream inspects data[0] to tell greyscale from colour and then
        # reshapes; rows of three keep it on the colour branch, where the
        # reshape reproduces the cached array exactly.
        return self._arr.reshape(-1, 3)


def _decode_like_upstream(path):
    '''Decode a file the way Pascal3DReal.__getitem__ does, so pixels match bit for bit.'''
    with open(path, "rb") as f:
        img_PIL = Image.open(f)
        img_PIL.convert("RGB")
        data = img_PIL.getdata()
        w, h = img_PIL.size
        if isinstance(data[0], int) or len(data[0]) == h * w:
            arr = np.array(data).reshape(h, w).reshape(h, w, 1).repeat(3, 2)
        else:
            arr = np.array(data).reshape(h, w, 3)
    return arr.astype(np.uint8)


class RawImageCache:
    '''Holds undecoded-from-disk inputs in RAM so augmentation can stay per-access.

    Pascal3D's augmentation is part of its camera solve: the flip rewrites the
    viewpoint angles, the jittered bounding box feeds get_desired_camera, and the
    rotation label falls out of the same computation. So a cached 224x224 crop is
    already an augmented sample and cannot be re-augmented. This caches one step
    earlier instead -- at the two file reads -- and lets __getitem__ run untouched,
    which keeps the label maths byte-identical to upstream.

    Pixels live in one flat uint8 tensor rather than a list of arrays: a list of
    ~10k numpy objects gets copied into every forked DataLoader worker as CPython
    touches their refcounts, while a single tensor buffer does not.
    '''

    def __init__(self, img_paths, annot_paths=(), workers: int = 8):
        self._index = {}
        self._annot = {}

        sizes = []
        for path in tqdm(img_paths, desc="Scanning image sizes"):
            with Image.open(path) as im:      # lazy: reads the header only
                w, h = im.size
            sizes.append((h, w))

        offset = 0
        for path, (h, w) in zip(img_paths, sizes):
            self._index[path] = (offset, h, w)
            offset += h * w * 3
        self._buf = torch.empty(offset, dtype=torch.uint8)

        def _fill(path):
            off, h, w = self._index[path]
            arr = _decode_like_upstream(path)
            self._buf[off:off + h * w * 3] = torch.from_numpy(np.ascontiguousarray(arr).reshape(-1))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(tqdm(pool.map(_fill, img_paths), total=len(img_paths), desc="Caching images"))

        for path in tqdm(list(annot_paths), desc="Caching annotations"):
            with open(path, "rb") as f:
                self._annot[path] = f.read()

    @classmethod
    def for_dataset(cls, dataset, workers: int = 8):
        real = dataset.real_dataset
        img_paths = list(real.img_paths)
        if getattr(dataset, "use_synth", False):
            img_paths.extend(dataset.synth_dataset.files)
        return cls(img_paths, real.annot_paths, workers=workers)

    @property
    def nbytes(self):
        return self._buf.nelement() + sum(len(v) for v in self._annot.values())

    def image(self, path):
        hit = self._index.get(path)
        if hit is None:
            return None
        off, h, w = hit
        return self._buf[off:off + h * w * 3].numpy().reshape(h, w, 3)

    def annotation(self, path):
        raw = self._annot.get(path)
        return None if raw is None else io.BytesIO(raw)

    def install(self):
        '''Serve pascal_dataset's two file reads from this cache.

        Patching the module's Image/loadmat names, rather than reimplementing
        __getitem__, is what guarantees the rotation labels stay identical.
        Paths this cache does not hold fall through to disk, so the validation
        split is unaffected.
        '''
        module = pascal_dataset_module

        if not getattr(module, "_raw_cache_installed", False):
            real_image, real_loadmat = module.Image, module.loadmat

            class _ImageProxy:
                def __getattr__(self, name):
                    return getattr(real_image, name)

                @staticmethod
                def open(f):
                    cache = getattr(module, "_raw_cache", None)
                    # Pascal3DReal passes a file object, Pascal3DSynth a path.
                    arr = cache.image(getattr(f, "name", f)) if cache is not None else None
                    return _CachedPIL(arr) if arr is not None else real_image.open(f)

            def _loadmat_proxy(path, *args, **kwargs):
                cache = getattr(module, "_raw_cache", None)
                buf = cache.annotation(path) if cache is not None else None
                return real_loadmat(path if buf is None else buf, *args, **kwargs)

            module.Image = _ImageProxy()
            module.loadmat = _loadmat_proxy
            module._raw_cache_installed = True

        module._raw_cache = self
        return self


class InMemoryDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        build_workers: int = 4,
        build_batch_size: int = 16,
        store_uint8: bool = True,
        img_key: str = "img",
        rot_key: str = "rot",
        use_multiprocessing: bool = False,
        n_draws: int = 1,
    ):
        self.base = base
        self.img_key = img_key
        self.rot_key = rot_key

        n = len(base)
        self.n = n
        self.n_draws = max(1, int(n_draws))

        sample = base[0]
        img0 = sample[img_key]
        rot0 = sample[rot_key]

        c, h, w = img0.shape
        rot_shape = rot0.shape

        total = n * self.n_draws
        if store_uint8:
            self.imgs = torch.empty((total, c, h, w), dtype=torch.uint8)
        else:
            self.imgs = torch.empty((total, c, h, w), dtype=torch.float32)

        self.targets = torch.empty((total, *rot_shape), dtype=torch.float32)

        self.store_uint8 = store_uint8

        gib = (self.imgs.element_size() * self.imgs.nelement()) / 2**30
        print(f"Caching {n} samples x {self.n_draws} draw(s) = {total} rows, {gib:.2f} GiB")

        # Each pass re-runs whatever randomness the base dataset applies in
        # __getitem__, so with use_warp the draws differ in pose as well as pixels.
        for draw in range(self.n_draws):
            desc = "Loading data into RAM"
            if self.n_draws > 1:
                desc += f" (draw {draw + 1}/{self.n_draws})"
            self._fill(base, draw * n, build_workers, build_batch_size, use_multiprocessing, desc)

    def _fill(self, base, offset, build_workers, build_batch_size, use_multiprocessing, desc):
        img_key, rot_key = self.img_key, self.rot_key
        store_uint8 = self.store_uint8
        n = self.n

        if use_multiprocessing:
            ctx = mp.get_context("spawn")
            chunks = _iter_chunks(n, build_batch_size)
            tasks = ((base, chunk, img_key, rot_key) for chunk in chunks)

            write_pos = offset
            with ctx.Pool(processes=max(1, build_workers)) as pool:
                for imgs, rots in tqdm(pool.imap(_load_chunk, tasks), total=(n + build_batch_size - 1) // build_batch_size, desc=desc):
                    bsz = imgs.shape[0]

                    if store_uint8:
                        if imgs.dtype != torch.uint8:
                            imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
                        else:
                            imgs_u8 = imgs
                        self.imgs[write_pos:write_pos + bsz].copy_(imgs_u8)
                    else:
                        self.imgs[write_pos:write_pos + bsz].copy_(imgs.to(torch.float32))

                    self.targets[write_pos:write_pos + bsz].copy_(rots.to(torch.float32))
                    write_pos += bsz
        else:
            loader = DataLoader(
                base,
                batch_size=build_batch_size,
                shuffle=False,
                num_workers=build_workers,
                pin_memory=False,
                persistent_workers=(build_workers > 0),
                prefetch_factor=4 if build_workers > 0 else None,
                collate_fn=_collate_keep(img_key, rot_key),
            )

            write_pos = offset
            for imgs, rots in tqdm(loader, desc=desc):
                bsz = imgs.shape[0]

                if store_uint8:
                    if imgs.dtype != torch.uint8:
                        imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
                    else:
                        imgs_u8 = imgs
                    self.imgs[write_pos:write_pos + bsz].copy_(imgs_u8)
                else:
                    self.imgs[write_pos:write_pos + bsz].copy_(imgs.to(torch.float32))

                self.targets[write_pos:write_pos + bsz].copy_(rots.to(torch.float32))
                write_pos += bsz

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.n_draws > 1:
            draw = int(torch.randint(self.n_draws, (1,)).item())
            idx = draw * self.n + idx

        x = self.imgs[idx]
        if self.store_uint8:
            x = x.to(torch.float32) / 255.0
        y = self.targets[idx]
        return {"img": x, "rot": y}


def create_dataloaders(config):
    if config.dataset == "dummynet":
        train_dataset = DummyPointCloudDataset(config, size=1000)
        val_dataset = DummyPointCloudDataset(config, size=100)
    elif not config.sanity_check:
        train = Pascal3D(config.path_to_datasets, train=True,
                         use_warp=config.use_warp, use_synth=config.use_synth)
        # Pascal3D asserts use_warp/use_synth are off for the test split.
        val = Pascal3D(config.path_to_datasets, train=False)

        if config.use_synth and len(train.synth_dataset.files) == 0:
            raise FileNotFoundError(
                "--use_synth found no RenderForCNN images under "
                f"{config.path_to_datasets}/syn_images_cropped_bkg_overlaid/<synset>/*/*.png"
            )
        if config.use_synth:
            available = len(train.synth_dataset.files)
            if 0 < config.max_synth < available:
                # Bound the pool before the cache sees it. Each epoch only draws
                # 3 * len(real) synthetic samples, so a capped pool still gives
                # every draw a fresh image.
                rng = np.random.default_rng(0)
                keep = sorted(rng.choice(available, size=config.max_synth, replace=False))
                train.synth_dataset.files = [train.synth_dataset.files[i] for i in keep]
            print(f"Synthetic pool: {len(train.synth_dataset.files)} of {available} renders, "
                  f"{3 * len(train.real_dataset)} drawn per epoch")

        num_builder = 4 if config.platform == "kaggle" else 2

        if config.ram_memory and config.raw_cache:
            # Cache the file reads and let Pascal3D augment on every access, so
            # augmentation is unlimited rather than a fixed pool of draws.
            cache = RawImageCache.for_dataset(train, workers=2 * num_builder).install()
            print(f"Raw cache: {cache.nbytes / 2**30:.2f} GiB held in RAM, "
                  f"augmentation runs per access")
            train_dataset = train
        elif config.ram_memory:
            if config.use_warp and config.cache_draws == 1:
                print("WARNING: --ram_memory caches one draw per image, which freezes the "
                      "augmentation --use_warp just enabled. Pass --cache_draws > 1 or "
                      "--raw_cache.")
            train_dataset = InMemoryDataset(train, build_workers=num_builder,
                                            use_multiprocessing=config.multiprocessing,
                                            n_draws=config.cache_draws)
        else:
            train_dataset = train
        # Validation stays deterministic: one draw, no augmentation.
        val_dataset = InMemoryDataset(val,build_workers=num_builder) if config.ram_memory else val
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(config)
    # The raw cache moves the warp into the workers, so it wants the full count.
    num_workers = 2 if (config.ram_memory and not config.raw_cache) else 4
    persistent_workers = (num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=True,persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False,persistent_workers=persistent_workers)
    return train_loader, val_loader


class DummyPointCloudDataset(Dataset):
    def __init__(self, config=None, path : str = None, size : int = 42, num_points=2048):
        super().__init__()
        path = config.path_to_datasets if config else path
        self.size = size
        create_technical_matrices(batch_size=self.size, device="cpu")
        self.base_path = pathlib.Path(path)
        self.meta = pd.read_csv(self.base_path / "metadata_modelnet10.csv")
        self.base_path /= "ModelNet10"
        self.point_cloud = MeshProcessor.to_point_cloud_array(file_path=self.base_path / "bed/train/bed_0001.off", num_points=num_points)
        self.num_points = num_points
        self.rotmats = project_to_orthogonal_manifold(torch.rand(self.size, 3, 3))

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return {
            "img" : torch.tensor(self.point_cloud, dtype=torch.float32) @ self.rotmats[i],
            "rot" : self.rotmats[i]
        }