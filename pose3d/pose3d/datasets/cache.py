"""RAM-resident datasets for Pascal3D: cached tensors and a raw file-read cache."""

import io
import multiprocessing as mp
import pathlib
from concurrent.futures import ThreadPoolExecutor

import image2sphere.pascal_dataset as pascal_dataset_module
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _collate_keep(img_key="img", rot_key="rot", cls_key=None):
    def _c(batch):
        imgs = torch.stack([b[img_key] for b in batch], dim=0)
        rots = torch.stack([b[rot_key] for b in batch], dim=0)
        clss = torch.stack([b[cls_key] for b in batch], dim=0) if cls_key is not None else None
        return imgs, rots, clss
    return _c


def _iter_chunks(n, chunk_size):
    for start in range(0, n, chunk_size):
        yield list(range(start, min(start + chunk_size, n)))


def _load_chunk(args):
    base, indices, img_key, rot_key, cls_key = args
    imgs = []
    rots = []
    clss = []
    for idx in indices:
        sample = base[idx]
        imgs.append(sample[img_key])
        rots.append(sample[rot_key])
        if cls_key is not None:
            clss.append(sample[cls_key])
    clss_t = torch.stack(clss, dim=0) if cls_key is not None else None
    return torch.stack(imgs, dim=0), torch.stack(rots, dim=0), clss_t


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
    '''Decode a file the way Pascal3DReal.__getitem__ does, so pixels match bit for bit.

    Upstream assembles the array from getdata(), which hands numpy one Python
    tuple per pixel: ~65 ms on a Pascal3D-sized JPEG against ~4 ms for asarray
    on the same file, and all of it under the GIL, so the fill pool below could
    never scale past one core. asarray reads the very same buffer in C.

    The branch reproduces what upstream actually sees rather than what it looks
    like it asks for: it discards the return of convert("RGB"), so getdata()
    runs on the file's own mode and a single-band image -- a palette one
    included -- comes out as its raw band repeated three times, never a palette
    lookup. Only 1- and 3-band files decode at all, here or upstream.
    '''
    with Image.open(path) as img_PIL:
        if img_PIL.mode == "1":
            # getdata() reports bilevel pixels as 0/255, the buffer holds 0/1.
            img_PIL = img_PIL.convert("L")
        arr = np.asarray(img_PIL)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, 2)
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
        cls_key: str = "cls",
        use_multiprocessing: bool = False,
        n_draws: int = 1,
        include_cls: bool = False,
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
        self.cls_key = cls_key if (include_cls and cls_key in sample) else None

        c, h, w = img0.shape
        rot_shape = rot0.shape

        total = n * self.n_draws
        if store_uint8:
            self.imgs = torch.empty((total, c, h, w), dtype=torch.uint8)
        else:
            self.imgs = torch.empty((total, c, h, w), dtype=torch.float32)

        self.targets = torch.empty((total, *rot_shape), dtype=torch.float32)

        self.eval_clss = None   # see set_eval_classes
        self.clss = None
        if self.cls_key is not None:
            cls0 = sample[self.cls_key]
            self.clss = torch.empty((total, *cls0.shape), dtype=cls0.dtype)

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
        img_key, rot_key, cls_key = self.img_key, self.rot_key, self.cls_key
        store_uint8 = self.store_uint8
        n = self.n

        if use_multiprocessing:
            ctx = mp.get_context("spawn")
            chunks = _iter_chunks(n, build_batch_size)
            tasks = ((base, chunk, img_key, rot_key, cls_key) for chunk in chunks)

            write_pos = offset
            with ctx.Pool(processes=max(1, build_workers)) as pool:
                for imgs, rots, clss in tqdm(pool.imap(_load_chunk, tasks), total=(n + build_batch_size - 1) // build_batch_size, desc=desc):
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
                    if self.clss is not None:
                        self.clss[write_pos:write_pos + bsz].copy_(clss)
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
                collate_fn=_collate_keep(img_key, rot_key, cls_key),
            )

            write_pos = offset
            for imgs, rots, clss in tqdm(loader, desc=desc):
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
                if self.clss is not None:
                    self.clss[write_pos:write_pos + bsz].copy_(clss)
                write_pos += bsz

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        blob = {"imgs": self.imgs, "targets": self.targets,
                "store_uint8": self.store_uint8, "n_draws": self.n_draws}
        if self.clss is not None:
            blob["clss"] = self.clss
        torch.save(blob, tmp)
        tmp.replace(path)   # a half-written file never looks like a finished cache

    @classmethod
    def load(cls, path, include_cls: bool = False):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        ds = cls.__new__(cls)
        ds.base = None
        ds.img_key, ds.rot_key = "img", "rot"
        ds.imgs, ds.targets, ds.store_uint8 = blob["imgs"], blob["targets"], blob["store_uint8"]
        ds.n_draws = int(blob.get("n_draws", 1))   # caches written before n_draws hold one draw
        ds.n = ds.imgs.shape[0] // ds.n_draws
        ds.eval_clss = None
        ds.clss = blob.get("clss")
        ds.cls_key = "cls" if (include_cls and ds.clss is not None) else None
        if include_cls and ds.clss is None:
            raise ValueError(f"{path} holds no class labels; rebuild the cache with fisher_prior on")
        if not include_cls:
            ds.clss = None
        return ds

    def set_eval_classes(self, classes):
        """Attach one class label per sample as batch["cls_eval"].

        A key of its own, not "cls": the training and validation losses hand "cls" to the model
        whenever it is present, and per-class metrics must not change what the model sees.
        """
        classes = torch.as_tensor(classes, dtype=torch.long).view(-1, 1)
        if len(classes) != self.n:
            raise ValueError(f"{len(classes)} class labels for {self.n} samples")
        self.eval_clss = classes

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
        out = {"img": x, "rot": y}
        if self.clss is not None:
            out["cls"] = self.clss[idx]
        if self.eval_clss is not None:
            out["cls_eval"] = self.eval_clss[idx % self.n]
        return out
