from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import Dataset
from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import DataLoader
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
    

class InMemoryDataset(Dataset):
    def __init__(self, base: Dataset, dtype_img=torch.uint8):
        self.base = base

        imgs = []
        targets = []

        for i in base:
            x, y = i["img"],i["rot"]           
            if isinstance(x, torch.Tensor):
                if dtype_img is not None and x.dtype != dtype_img:
                    if x.dtype.is_floating_point:
                        x = (x.clamp(0, 1) * 255).to(torch.uint8)
                    else:
                        x = x.to(dtype_img)
            imgs.append(x.cpu())
            targets.append(y)

        self.imgs = imgs
        self.targets = targets

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = self.imgs[idx]
        y = self.targets[idx]

        if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
            x = x.float().div(255.0)

        return {"img" : x, "rot" : y}



def create_dataloaders(config):

    if not config.sanity_check:
        train_dataset = InMemoryDataset(Pascal3D(config.path_to_datasets, True)) if config.ram_memory else Pascal3D(config.path_to_datasets, True)
        val_dataset = InMemoryDataset(Pascal3D(config.path_to_datasets, False)) if config.ram_memory else Pascal3D(config.path_to_datasets, False)
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(config)
    num_workers = 0 if config.ram_memory else 4
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=True,persistent_workers=True,prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False,persistent_workers=True,prefetch_factor=4)
    return train_loader, val_loader
