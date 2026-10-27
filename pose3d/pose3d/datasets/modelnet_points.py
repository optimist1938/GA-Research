"""ModelNet10 as point clouds under random SO(3) poses: the ModelNet10-SO(3) task with the geometry as input.

Same objects, canonical frames and split as ModelNet10-SO(3) (Liao et al. 2019; image2sphere's benchmark): the
manually aligned ModelNet10 meshes, 3,991 train / 908 test models in 10 classes. Instead of a rendered image the
input is a point cloud sampled from the mesh surface and turned by the pose R (x = R x_canonical); the label is R.
Symmetric classes (bathtub, desk, night stand, table) keep their ambiguous poses with a single label, as there.

Layout under cfg.run.path_to_datasets: the Princeton release, ModelNet10/<class>/{train,test}/<class>_<id>.off
(e.g. Kaggle's balraj98/modelnet10-princeton-3d-object-dataset), found anywhere below the root. The meshes are
sampled once (area-weighted, `pool` points per model, fixed seed), centred on their bounding box and scaled into
the unit ball, then cached as an .npz next to the code (or in `cache_dir`).

Train items: a fresh uniform rotation and a fresh subset of `n_points` points at every read; `repeats` draws
per model per epoch. Test items: every
model under `test_rotations` fixed rotations (ModelNet10-SO(3) has 4 test views per model) with fixed subsets.
"""

import glob
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ("bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet")


def read_off(path):
    """Vertices (V, 3) and triangles (F, 3) of an OFF mesh (polygons fan-triangulated). Tolerates ModelNet's
    header glitch, where the counts follow 'OFF' on the same line."""
    with open(path) as f:
        tokens = f.read().split()
    head = tokens[0]
    if head == "OFF":
        i = 1
    elif head.startswith("OFF"):
        tokens = [head[3:]] + tokens[1:]
        i = 0
    else:
        raise ValueError(f"{path}: not an OFF file")
    nv, nf = int(tokens[i]), int(tokens[i + 1])
    i += 3
    verts = np.asarray(tokens[i:i + 3 * nv], dtype=np.float64).reshape(nv, 3)
    i += 3 * nv
    rest = tokens[i:]
    if len(rest) >= 4 * nf:
        tri = np.asarray(rest[:4 * nf], dtype=np.int64).reshape(nf, 4)
        if (tri[:, 0] == 3).all():                                   # all triangles: the common case, vectorised
            return verts, tri[:, 1:]
    faces = []
    for _ in range(nf):
        k = int(tokens[i])
        idx = [int(v) for v in tokens[i + 1:i + 1 + k]]
        faces.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1))
        i += 1 + k
    return verts, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def sample_surface(verts, faces, n, rng):
    """n points uniformly on the surface (area-weighted triangles, uniform barycentric coordinates)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])


def normalize(points):
    """Centre on the bounding box and scale into the unit ball (pose-independent up to the rotation)."""
    centre = 0.5 * (points.min(0) + points.max(0))
    p = points - centre
    return p / np.linalg.norm(p, axis=1).max()


def _sample_one(args):
    path, pool, seed = args
    verts, faces = read_off(path)
    return normalize(sample_surface(verts, faces, pool, np.random.default_rng(seed))).astype(np.float32)


def build_cache(root, split, out_path, pool=2048, workers=16):
    files = []
    for c, name in enumerate(CLASSES):
        found = sorted(glob.glob(os.path.join(root, "**", "ModelNet10", name, split, "*.off"), recursive=True))
        files += [(f, c) for f in found]
    if not files:
        raise FileNotFoundError(f"no ModelNet10/<class>/{split}/*.off below {root}")
    jobs = [(f, pool, 1000003 * (split == "test") + i) for i, (f, _) in enumerate(files)]
    if "fork" in mp.get_all_start_methods():                         # parsing is pure Python: processes
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            points = list(ex.map(_sample_one, jobs, chunksize=8))
    else:
        points = [_sample_one(j) for j in jobs]
    np.savez(out_path, points=np.stack(points), cls=np.array([c for _, c in files]),
             names=np.array([os.path.basename(f)[:-4] for f, _ in files]))


def random_rotations(n, generator=None):
    """Uniform rotations from unit quaternions."""
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator, dtype=torch.float64), dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(n, 3, 3)
    return R.float()


class ModelNetPoints(Dataset):
    num_classes = len(CLASSES)
    class_names = CLASSES

    def __init__(self, root, train: bool, n_points: int = 256, test_rotations: int = 4, pool: int = 2048,
                 cache_dir: str = None, repeats: int = 1):
        split = "train" if train else "test"
        cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"modelnet10_points_{split}_{pool}.npz")
        if not os.path.exists(path):
            build_cache(root, split, path, pool)
        data = np.load(path)
        self.points = torch.from_numpy(data["points"])            # (M, pool, 3), canonical frame
        self.cls = torch.from_numpy(data["cls"]).long()
        self.train, self.n_points, self.repeats = train, n_points, max(1, int(repeats))
        if not train:
            gen = torch.Generator().manual_seed(0)
            m = len(self.points)
            self.test_rot = random_rotations(m * test_rotations, gen)
            self.test_idx = torch.stack([torch.randperm(pool, generator=gen)[:n_points]
                                         for _ in range(m * test_rotations)])
            self.test_rotations = test_rotations

    def __len__(self):
        return len(self.points) * self.repeats if self.train else len(self.test_rot)

    def __getitem__(self, i):
        if self.train:
            model = i % len(self.points)
            R = random_rotations(1)[0]
            idx = torch.randperm(self.points.shape[1])[:self.n_points]
        else:
            model = i // self.test_rotations
            R, idx = self.test_rot[i], self.test_idx[i]
        x = self.points[model, idx] @ R.T                          # x = R x_canonical
        return dict(img=x, rot=R, cls=self.cls[model:model + 1])
