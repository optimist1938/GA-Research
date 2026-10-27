"""SYMSOL I as point clouds: the tetrahedron, cube, icosahedron, cone and cylinder of Murphy et al. (2021), given
as points sampled from their surfaces instead of rendered images.

Each shape is centred and scaled into the unit ball in its canonical frame, where its rotation symmetries form a
group G (12 / 24 / 60 elements; the cone's rotations about e3 and the cylinder's, also turned end over end, at
1 degree as in SYMSOL: 360 / 720). A cloud is x = R x_canonical with R uniform on SO(3); every R g, g in G, is an
equally correct label. Train items draw fresh points, a fresh pose and a fresh equivalent at every read (as
datasets/symsol.py); test items are fixed (seeded per index) and carry all equivalents ("rots_eq", padded to 720,
"n_eq") for the nearest-equivalent error and the log-likelihood. No data files: everything is generated.
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from pose3d.datasets.modelnet_points import random_rotations, sample_surface
from pose3d.datasets.symsol import MAX_EQUIVALENTS, SHAPE_SETS

PHI = (1 + 5 ** 0.5) / 2


def _axis_angle(axis, angle):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _closure(generators):
    group = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for g in frontier:
            for h in generators:
                m = g @ h
                if not any(np.abs(m - x).max() < 1e-8 for x in group):
                    group.append(m)
                    new.append(m)
        frontier = new
    return np.stack(group)


def _hull_faces(verts, edge):
    """Triangles of a regular polyhedron: vertex triples at mutual distance `edge`, oriented outwards."""
    n, faces = len(verts), []
    d = np.linalg.norm(verts[:, None] - verts[None], axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if all(abs(x - edge) < 1e-6 for x in (d[i, j], d[i, k], d[j, k])):
                    faces.append([i, j, k])
    return np.array(faces)


def _polyhedra():
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    ico = []
    for a in (-1, 1):
        for b in (-PHI, PHI):
            ico += [[0, a, b], [a, b, 0], [b, 0, a]]
    ico = np.array(ico, dtype=np.float64)
    cube_faces = []
    for axis in range(3):
        for sign in (-1, 1):
            quad = [i for i, v in enumerate(cube) if v[axis] == sign]
            # the 4 vertices of a face: split into two triangles across a diagonal
            v = cube[quad]
            far = max(range(1, 4), key=lambda k: np.linalg.norm(v[k] - v[0]))
            rest = [k for k in range(1, 4) if k != far]
            cube_faces += [[quad[0], quad[rest[0]], quad[far]], [quad[0], quad[rest[1]], quad[far]]]
    groups = {
        "tet": _closure([_axis_angle([1, 1, 1], 2 * math.pi / 3), _axis_angle([0, 0, 1], math.pi)]),
        "cube": _closure([_axis_angle([0, 0, 1], math.pi / 2), _axis_angle([1, 0, 0], math.pi / 2)]),
        "icosa": _closure([np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
                           _axis_angle([0, 0, 1], math.pi), _axis_angle([0, 1, PHI], 2 * math.pi / 5)]),
    }
    meshes = {"tet": (tet / np.sqrt(3), _hull_faces(tet, 2 * np.sqrt(2))),
              "cube": (cube / np.sqrt(3), np.array(cube_faces)),
              "icosa": (ico / np.sqrt(1 + PHI ** 2), _hull_faces(ico, 2.0))}
    return meshes, groups


def _revolution_groups():
    rz = np.stack([_axis_angle([0, 0, 1], math.radians(k)) for k in range(360)])
    flip = _axis_angle([1, 0, 0], math.pi)
    return {"cone": rz, "cyl": np.concatenate([rz, rz @ flip])}


def _sample_cone(n, rng):
    """Apex (0, 0, 1), base radius 1 at z = -1, scaled into the unit ball (by sqrt 2)."""
    lateral, base = math.pi * math.sqrt(5), math.pi
    on_side = rng.random(n) < lateral / (lateral + base)
    theta = rng.random(n) * 2 * math.pi
    s = np.sqrt(rng.random(n))                      # area-uniform: radius grows linearly from the apex
    r = np.where(on_side, s, np.sqrt(rng.random(n)))
    z = np.where(on_side, 1 - 2 * s, -1.0)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], -1) / math.sqrt(2)


def _sample_cylinder(n, rng):
    """Radius 1, z in [-1, 1], scaled into the unit ball (by sqrt 2)."""
    lateral, caps = 4 * math.pi, 2 * math.pi
    on_side = rng.random(n) < lateral / (lateral + caps)
    theta = rng.random(n) * 2 * math.pi
    r = np.where(on_side, 1.0, np.sqrt(rng.random(n)))
    z = np.where(on_side, 2 * rng.random(n) - 1, np.where(rng.random(n) < 0.5, -1.0, 1.0))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], -1) / math.sqrt(2)


class SymsolPoints(Dataset):
    def __init__(self, train: bool, n_points: int = 512, views: int = 4500, set_number: int = 1,
                 fixed: bool = False):
        if set_number != 1:
            raise ValueError("SymsolPoints has SYMSOL I only")
        self.shapes = SHAPE_SETS[1]
        # fixed=True: a training set of `views` fixed clouds per shape (fixed pose and points, seeded per index),
        # read again every epoch with a fresh equivalent label -- like SYMSOL's fixed rendered views, so the
        # "10 %" setting (4,500 views per shape) means the same thing for images and points.
        self.train, self.n_points, self.views, self.fixed = train, n_points, views, fixed
        meshes, groups = _polyhedra()
        groups.update(_revolution_groups())
        self.meshes = meshes
        self.groups = [torch.from_numpy(groups[s]).float() for s in self.shapes]
        self.cls = torch.arange(len(self.shapes)).repeat_interleave(views)

    def __len__(self):
        return len(self.cls)

    def sample(self, c, n, rng):
        name = self.shapes[c]
        if name == "cone":
            return _sample_cone(n, rng)
        if name == "cyl":
            return _sample_cylinder(n, rng)
        verts, faces = self.meshes[name]
        return sample_surface(verts, faces, n, rng)

    def __getitem__(self, i):
        c = int(self.cls[i])
        G = self.groups[c]
        if self.train and self.fixed:
            seed = 104729 * i + 17                                  # disjoint from the test seeds (7919 i + 1)
            gen = torch.Generator().manual_seed(seed)
            rng = np.random.default_rng(seed)
            R = random_rotations(1, gen)[0]
            label = R @ G[torch.randint(len(G), ()).item()]
        elif self.train:
            rng = np.random.default_rng(torch.randint(2 ** 62, ()).item())
            R = random_rotations(1)[0]
            label = R @ G[torch.randint(len(G), ()).item()]
        else:
            gen = torch.Generator().manual_seed(7919 * i + 1)
            rng = np.random.default_rng(7919 * i + 1)
            R = random_rotations(1, gen)[0]
            label = R
        x = torch.from_numpy(self.sample(c, self.n_points, rng)).float() @ R.T
        cls = self.cls[i:i + 1]
        if self.train:
            return dict(img=x, rot=label, cls_eval=cls)
        eq = R @ G                                                  # every correct label
        padded = eq.new_zeros(MAX_EQUIVALENTS, 3, 3)
        padded[:len(eq)] = eq
        return dict(img=x, rot=R, cls_eval=cls, rots_eq=padded, n_eq=torch.tensor(len(eq)))
