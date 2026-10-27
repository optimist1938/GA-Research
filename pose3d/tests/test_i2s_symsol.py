"""The Image2Sphere baseline (I2SReal) on SYMSOL: its lazily built evaluation grid, the nearest-equivalent
error path and the grid log-likelihood that mirrors upstream's evaluate_ll."""

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pose3d.datasets.symsol import Symsol
from pose3d.engine.grid_model_likelihood import LOG_VOLUME, grid_model_log_likelihood, nearest_grid_index
from pose3d.engine.metrics import calculate_evaluation_metrics
from pose3d.models.i2s_real import I2SReal
from tests.test_symsol import _fake_symsol


class _Cfg:
    class run:
        platform = "colab"

    class features:
        fixed_val_noise = False

    device = torch.device("cpu")
    rank = 0
    world_size = 1


def _model(level=2):
    torch.manual_seed(0)
    return I2SReal(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=level).eval()


def test_eval_grid_is_built_lazily_and_matches_upstream():
    from image2sphere import so3_utils
    from e3nn import o3

    model = _model(level=2)
    assert model._grid is None
    wig, rots = model._eval_grid("cpu")
    a, b, g = so3_utils.so3_healpix_grid(rec_level=2)
    assert rots.shape == (a.numel(), 3, 3) and wig.shape == (455, a.numel())
    torch.testing.assert_close(rots, o3.angles_to_matrix(a, b, g))
    torch.testing.assert_close(wig, so3_utils.flat_wigner(6, a, b, g).T)
    assert model._eval_grid("cpu") is model._grid   # cached


def test_predict_returns_grid_rotations_and_logits_match():
    model = _model(level=2)
    img = torch.rand(2, 3, 224, 224)
    wig, rots = model._eval_grid("cpu")
    logits = model.grid_logits(img)
    assert logits.shape == (2, rots.shape[0])
    # Upstream's orthographic projection draws a random subset of sphere points at every forward pass (also in
    # eval mode), so two passes differ slightly: seed them alike to compare predict with the argmax of the logits.
    torch.manual_seed(1)
    pred = model.predict(img)
    torch.manual_seed(1)
    torch.testing.assert_close(pred, rots[model.grid_logits(img).argmax(1)])
    torch.testing.assert_close(pred.transpose(-1, -2) @ pred, torch.eye(3).expand(2, 3, 3), atol=1e-5, rtol=0)


def test_nearest_grid_index():
    model = _model(level=2)
    _, rots = model._eval_grid("cpu")
    idx = torch.tensor([5, 1000, 3])
    assert nearest_grid_index(rots[idx], rots).tolist() == idx.tolist()


class _Delta(torch.nn.Module):
    """A grid model that puts all of its mass on the cell of the label it is told (every image the same)."""

    def __init__(self, base, cell):
        super().__init__()
        self.base, self.cell = base, cell

    def _eval_grid(self, device):
        return self.base._eval_grid(device)

    def grid_logits(self, img):
        _, rots = self._eval_grid(img.device)
        out = torch.full((img.shape[0], rots.shape[0]), -1e4)
        out[:, self.cell] = 0.0
        return out


def test_grid_log_likelihood_of_a_delta_is_the_cell_density(tmp_path):
    root = _fake_symsol(str(tmp_path), n_views=2)
    test = Symsol(root, train=False)
    base = _model(level=2)
    _, rots = base._eval_grid("cpu")
    # Relabel every test image so that all of its equivalents are grid rotation 7: the mass is in that cell.
    for r in test.rots:
        r[:] = rots[7]
    loader = DataLoader(test, batch_size=4, shuffle=False)
    ll, cls = grid_model_log_likelihood(_Delta(base, 7), loader, "cpu", n_equivalents=3, seed=0)
    assert ll.shape == (len(test),) and sorted(set(cls.tolist())) == [0, 1, 2, 3, 4]
    n = rots.shape[0]
    np.testing.assert_allclose(ll, math.log(n) - LOG_VOLUME, atol=1e-5)   # the whole cell density, N / pi^2
    ll_miss, _ = grid_model_log_likelihood(_Delta(base, 8), loader, "cpu", n_equivalents=3, seed=0)
    assert (ll_miss < -100).all()                                          # the label's cell is empty


def test_uniform_grid_model_scores_the_uniform_density(tmp_path):
    root = _fake_symsol(str(tmp_path), n_views=1)
    test = Symsol(root, train=False)
    base = _model(level=2)

    class Uniform(_Delta):
        def grid_logits(self, img):
            _, rots = self._eval_grid(img.device)
            return torch.zeros(img.shape[0], rots.shape[0])

    ll, _ = grid_model_log_likelihood(Uniform(base, 0), DataLoader(test, batch_size=5), "cpu", n_equivalents=3)
    np.testing.assert_allclose(ll, -LOG_VOLUME, atol=1e-5)   # log(1/pi^2) = -2.29


def test_symsol_evaluation_uses_the_nearest_equivalent(tmp_path):
    root = _fake_symsol(str(tmp_path), n_views=1)
    test = Symsol(root, train=False)
    torch.manual_seed(0)
    fixed = _model(level=2)._eval_grid("cpu")[1][torch.randperm(5000)[:5]]   # five grid rotations as predictions

    class Fixed(torch.nn.Module):
        def predict(self, img):
            return fixed[: img.shape[0]]

    err, cls = calculate_evaluation_metrics(Fixed(), DataLoader(test, batch_size=5), _Cfg, return_classes=True)
    assert err.shape == (5,) and cls.tolist() == [0, 1, 2, 3, 4]
    for i in range(5):
        item = test[i]
        eq = item["rots_eq"][: int(item["n_eq"])]
        cos = ((fixed[i].T @ eq).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        nearest = math.degrees(math.acos(float(cos.clamp(-1, 1).max())))
        to_label = math.degrees(math.acos(float(cos[0].clamp(-1, 1))))
        assert abs(err[i] - nearest) < 1e-3
        assert err[i] <= to_label + 1e-3          # the labelled pose is one of the equivalents, never nearer
    assert (err < 180).all() and (err >= 0).all()
