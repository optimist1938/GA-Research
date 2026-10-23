"""SYMSOL (Murphy et al., ICML 2021): renders of symmetric solids, every image labelled with all of its
equivalent poses (12 for the tetrahedron, 24 cube, 60 icosahedron, 360 at 1 degree for the cone, 720 for the
cylinder: each of those also turned end over end).

Layout under cfg.run.path_to_datasets, as in the official symsol_dataset.zip (and image2sphere):
    symsol/{train,test}/rotations.npz      {shape: (n_views, n_equivalents, 3, 3)}, index 0 = rendered pose
    symsol/{train,test}/images/<shape>_<index:05d>.png      224 x 224 RGB

Unlike image2sphere.dataset.SymsolDataset (one label drawn per access, so a RAM cache freezes it, and the
test error is taken to one random equivalent), this keeps the images in RAM as uint8, draws a fresh
equivalent every time a training image is read, and gives the test set every equivalent ("rots_eq",
padded to MAX_EQUIVALENTS, with "n_eq") for the nearest-equivalent error and the log-likelihood.
`num_views` keeps the first views of every shape (Howell et al.'s 10 % setting is 4,500 of 45,000).

Poses are converted to this repository's camera frame (e1 image columns, e2 rows down, e3 optical axis,
models/so3fourier_head.py), so a 90 degree turn of the image is R -> R_z(-90) R as the equivariant heads
assume. SYMSOL's own frame is fixed by TO_CAMERA, which tests/test_symsol.py checks on the images.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SHAPE_SETS = {
    1: ("tet", "cube", "icosa", "cone", "cyl"),
    2: ("sphereX",),
    3: ("cylO",),
    4: ("tetX",),
}
MAX_EQUIVALENTS = 720

# R_camera = TO_CAMERA @ R_symsol. SYMSOL's poses already are in this frame: the projected axis of the
# cylinder lines up with its silhouette (0.998, against 0.03 for the other signed axis permutations) and the
# cone's apex points along +R e3 (0.991; -0.99 for a 180 degree turn, 0.00 for 90 degrees), on 300 test views.
TO_CAMERA = np.eye(3)


def to_camera_frame(rots):
    return np.einsum("ij,...jk->...ik", TO_CAMERA, rots).astype(np.float32)


def _read_png(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


class Symsol(Dataset):
    def __init__(self, root: str, train: bool, set_number: int = 1, num_views=None, workers: int = 16,
                 train_equivalents: bool = False):
        self.train = train
        self.train_equivalents = train_equivalents   # training items also carry rots_eq / n_eq (nearest coupling)
        self.shapes = SHAPE_SETS[set_number]
        path = os.path.join(root, "symsol", "train" if train else "test")
        rotations = np.load(os.path.join(path, "rotations.npz"))
        self.rots, files, cls = [], [], []
        for c, name in enumerate(self.shapes):
            r = rotations[name][:num_views]
            self.rots.extend(torch.from_numpy(to_camera_frame(r)))
            files += [os.path.join(path, "images", f"{name}_{i:05d}.png") for i in range(len(r))]
            cls += [c] * len(r)
        with ThreadPoolExecutor(workers) as pool:
            images = list(pool.map(_read_png, files))
        self.images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()   # (N, 3, H, W) uint8
        self.cls = torch.tensor(cls, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        rots = self.rots[i]
        img = self.images[i].float() / 255.0
        cls = self.cls[i:i + 1]
        if self.train and not self.train_equivalents:
            # A fresh equivalent every read: the flow learns the whole symmetric distribution.
            return dict(img=img, rot=rots[torch.randint(len(rots), ()).item()], cls_eval=cls)
        padded = rots.new_zeros(MAX_EQUIVALENTS, 3, 3)
        padded[:len(rots)] = rots
        return dict(img=img, rot=rots[0], cls_eval=cls, rots_eq=padded, n_eq=torch.tensor(len(rots)))
