from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import Dataset
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

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

class InMemoryDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        build_workers: int = 4,
        build_batch_size: int = 16,
        store_uint8: bool = True,
        img_key: str = "img",
        rot_key: str = "rot",
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
        for imgs, rots in tqdm(loader):
            bsz = imgs.shape[0]

            if store_uint8:
                if imgs.dtype != torch.uint8:
                    imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
                else:
                    imgs_u8 = imgs
                self.imgs[write_pos:write_pos+bsz].copy_(imgs_u8)
            else:
                self.imgs[write_pos:write_pos+bsz].copy_(imgs.to(torch.float32))

            self.targets[write_pos:write_pos+bsz].copy_(rots.to(torch.float32))
            write_pos += bsz

        self.store_uint8 = store_uint8

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        x = self.imgs[idx]
        if self.store_uint8:
            x = x.to(torch.float32) / 255.0
        y = self.targets[idx]
        return {"img": x, "rot": y}



def create_dataloaders(config):

    if not config.sanity_check:
        train = Pascal3D(config.path_to_datasets, train=True)
        val = Pascal3D(config.path_to_datasets, train=False)
        train_dataset = InMemoryDataset(train) if config.ram_memory else train
        val_dataset = InMemoryDataset(val) if config.ram_memory else val
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(config)
    num_workers = 2 if config.ram_memory else 4
    persistent_workers = (num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=True,persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False,persistent_workers=persistent_workers)
    return train_loader, val_loader
