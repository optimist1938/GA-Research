from image2sphere.pascal_dataset import Pascal3D
import torch
from image2sphere.pascal_dataset import Pascal3D
from tqdm import tqdm
import multiprocessing as mp
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
    ):
        self.base = base
        self.img_key = img_key
        self.rot_key = rot_key

        n = len(base)

        sample = base[0]
        img0 = sample[img_key]
        rot0 = sample[rot_key]

        c, h, w = img0.shape
        rot_shape = rot0.shape

        if store_uint8:
            self.imgs = torch.empty((n, c, h, w), dtype=torch.uint8)
        else:
            self.imgs = torch.empty((n, c, h, w), dtype=torch.float32)

        self.targets = torch.empty((n, *rot_shape), dtype=torch.float32)

        self.store_uint8 = store_uint8

        if use_multiprocessing:
            ctx = mp.get_context("spawn")
            chunks = _iter_chunks(n, build_batch_size)
            tasks = ((base, chunk, img_key, rot_key) for chunk in chunks)

            write_pos = 0
            with ctx.Pool(processes=max(1, build_workers)) as pool:
                for imgs, rots in tqdm(pool.imap(_load_chunk, tasks), total=(n + build_batch_size - 1) // build_batch_size, desc="Loading data into RAM"):
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

            write_pos = 0
            for imgs, rots in tqdm(loader, desc="Loading data into RAM"):
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
        return self.imgs.shape[0]

    def __getitem__(self, idx):
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
        train = Pascal3D(config.path_to_datasets, train=True)
        val = Pascal3D(config.path_to_datasets, train=False)
        num_builder = 4 if config.platform == "kaggle" else 2
        train_dataset = InMemoryDataset(train,build_workers=num_builder, use_multiprocessing=config.multiprocessing) if config.ram_memory else train
        val_dataset = InMemoryDataset(val,build_workers=num_builder) if config.ram_memory else val
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(config)
    num_workers = 2 if config.ram_memory else 4
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