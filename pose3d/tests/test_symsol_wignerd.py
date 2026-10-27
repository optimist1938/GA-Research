import math
import os

import numpy as np
import pytest
import torch
from PIL import Image

import pose3d.geometry.matrix as matrix_flow
from pose3d.datasets.symsol import SHAPE_SETS, Symsol

pytest.importorskip("image2sphere")
SHAPES = {"tet": 12, "cube": 24, "icosa": 60, "cone": 360, "cyl": 720}


def _fake_symsol(root, n_views=3, size=64):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        path = os.path.join(root, "symsol", split)
        os.makedirs(os.path.join(path, "images"))
        rots = {}
        for name, k in SHAPES.items():
            rots[name] = matrix_flow.exp_map(torch.tensor(rng.normal(size=(n_views, k, 3)))).numpy()
            for i in range(n_views):
                Image.fromarray(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)).save(os.path.join(path, "images", f"{name}_{i:05d}.png"))
        np.savez(os.path.join(path, "rotations.npz"), **rots)
    return root


class _OracleGrid:
    """Stands in for WignerD: puts all the probability on the grid cell nearest to the first equivalent of each image."""

    def __init__(self, rots, first):
        self.rots, self.first, self.i = rots, first, 0

    def eval(self):
        return self

    def _eval_grid(self, device):
        return None, self.rots

    def grid_logits(self, img):
        out = torch.zeros(img.shape[0], self.rots.shape[0])
        for j in range(img.shape[0]):
            out[j, int((self.first[self.i].reshape(1, 9) @ self.rots.reshape(-1, 9).T).argmax())] = 1e3
            self.i += 1
        return out


def test_grid_log_likelihood_cap_and_calibration(tmp_path):
    from e3nn import o3
    from image2sphere import so3_utils
    from pose3d.engine.symsol_ll import LOG_VOLUME, summarize, wigner_grid_log_likelihood

    test = Symsol(_fake_symsol(str(tmp_path)), train=False)
    rots = o3.angles_to_matrix(*so3_utils.so3_healpix_grid(rec_level=2))
    order = [test[i]["rots_eq"][0] for c in range(5) for i in (test.cls == c).nonzero().flatten().tolist()]
    ll, classes, cap = wigner_grid_log_likelihood(_OracleGrid(rots, order), test, "cpu", n_equivalents=4, temperatures=(1.0, 4.0), batch=2, log=lambda s: None)
    assert ll.shape == (2, 15) and classes.tolist() == sorted(classes.tolist())
    assert (ll <= cap[None] + 1e-5).all()                       # never above log(N/pi^2) - log(#equivalents)
    s = summarize(ll, classes, temperatures=(1.0, 4.0), n_cal=1)
    assert math.isfinite(s["log_likelihood"]) and s["log_likelihood_temperature"] in (1.0, 4.0)
    assert abs(math.log(rots.shape[0]) - LOG_VOLUME - cap.max()) < 1e-4 or cap.max() < math.log(rots.shape[0]) - LOG_VOLUME


def test_wignerd_grid_logits_peak_at_the_label():
    import torch
    from pose3d.models.wignerd import WignerD
    from scipy.spatial.transform import Rotation

    m = WignerD(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=2).eval()
    wig, rots = m._eval_grid("cpu")
    R = rots[[5, 100]]
    psi = m.target(R)                                            # a perfect regressor's output
    assert ((psi @ wig).argmax(1) == torch.tensor([5, 100])).all()
