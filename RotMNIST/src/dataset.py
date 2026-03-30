import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


class RotMNISTInternal(Dataset):

    def __init__(self, root, train=True, download=True, seed=42):
        self.mnist = datasets.MNIST(root, train=train, download=download,
                                    transform=None)
        rng = np.random.RandomState(seed + (0 if train else 1))
        self.angles_deg = rng.uniform(0, 360, size=len(self.mnist)).astype(np.float32)
        self.to_tensor = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img_pil, _ = self.mnist[idx]         
        angle = self.angles_deg[idx]

        img_np = np.array(img_pil, dtype=np.uint8)  

        rows = np.any(img_np > 0, axis=1)
        cols = np.any(img_np > 0, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            digit_crop = img_np[rmin:rmax + 1, cmin:cmax + 1]

            pil_crop = Image.fromarray(digit_crop, mode='L')
            rotated = pil_crop.rotate(angle, expand=True, fillcolor=0)
            rc = np.array(rotated, dtype=np.uint8)
        else:
            rc = img_np

        canvas = np.zeros((28, 28), dtype=np.uint8)
        rh, rw = rc.shape
        r0 = max(0, 14 - rh // 2)
        c0 = max(0, 14 - rw // 2)
        r1 = min(28, r0 + rh)
        c1 = min(28, c0 + rw)
        canvas[r0:r1, c0:c1] = rc[:r1 - r0, :c1 - c0]

        img_tensor = self.to_tensor(Image.fromarray(canvas, mode='L'))  # (1, 28, 28)

        angle_rad = np.deg2rad(angle)
        label = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)], dtype=torch.float32)

        return img_tensor, label


def create_dataloaders(config):
    train_ds = RotMNISTInternal(config.data_dir, train=True, download=True)
    val_ds = RotMNISTInternal(config.data_dir, train=False, download=True)

    if config.sanity_check:
        train_ds = Subset(train_ds, range(config.batch_size * 4))
        val_ds = Subset(val_ds, range(config.batch_size * 2))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
