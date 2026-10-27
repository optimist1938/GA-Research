"""ModelNet10-SO(3) (image2sphere's dataset class, used as-is) and SYMSOL (datasets/symsol.py) loading.

Both return the same {img, rot, cls} dict shape as Pascal3D, at a fixed 224x224, so no
resizing/collation changes are needed elsewhere. `cfg.run.path_to_datasets` must contain a
`modelnet10/` (modelnet10_{train,test}.npz) or `symsol/` ({train,test}/rotations.npz + images/)
subfolder depending on cfg.run.dataset. Both are RAM-resident once built: ModelNet10Dataset loads its
whole .npz in __init__, Symsol decodes every PNG once.
"""

import time

from image2sphere.dataset import ModelNet10Dataset

from pose3d.datasets.symsol import Symsol


def create_benchmark_datasets(cfg):
    path = cfg.run.path_to_datasets
    if cfg.run.dataset == "modelnet10":
        return ModelNet10Dataset(path, train=True), ModelNet10Dataset(path, train=False)
    if cfg.run.dataset == "symsol_points":
        from pose3d.datasets.symsol_points import SymsolPoints
        d = cfg.data
        return (SymsolPoints(True, n_points=d.points_n, views=d.symsol_train_views or 4500, fixed=d.points_fixed_train),
                SymsolPoints(False, n_points=d.points_n, views=d.symsol_test_views or 1000))
    if cfg.run.dataset == "modelnet10_points":
        from pose3d.datasets.modelnet_points import ModelNetPoints
        d, t0 = cfg.data, time.time()
        train = ModelNetPoints(path, train=True, n_points=d.points_n, repeats=d.points_repeats)
        val = ModelNetPoints(path, train=False, n_points=d.points_n, test_rotations=d.points_test_rotations)
        print(f"[timing] modelnet10_points: {len(train)} train models / {len(val)} test clouds in "
              f"{time.time() - t0:.1f}s")
        return train, val
    # Our own SYMSOL loader: it keeps every equivalent pose (datasets/symsol.py).
    d = cfg.data
    t0 = time.time()
    train = Symsol(path, train=True, set_number=d.symsol_set, num_views=d.symsol_train_views,
                   train_equivalents=d.symsol_nearest_coupling)
    val = Symsol(path, train=False, set_number=d.symsol_set, num_views=d.symsol_test_views)
    print(f"[timing] symsol: {len(train)} train / {len(val)} test images in {time.time() - t0:.1f}s")
    return train, val
