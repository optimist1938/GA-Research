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
    # Our own SYMSOL loader: it keeps every equivalent pose (datasets/symsol.py).
    d = cfg.data
    t0 = time.time()
    train = Symsol(path, train=True, set_number=d.symsol_set, num_views=d.symsol_train_views,
                   train_equivalents=d.symsol_nearest_coupling)
    val = Symsol(path, train=False, set_number=d.symsol_set, num_views=d.symsol_test_views)
    print(f"[timing] symsol: {len(train)} train / {len(val)} test images in {time.time() - t0:.1f}s")
    return train, val
