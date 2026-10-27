"""engine/grid_likelihood.py: Image2Sphere's grid, the cell assignment and the grid log-likelihood."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import pose3d.geometry.matrix as matrix_flow
from pose3d.engine.grid_likelihood import (
    LOG_VOLUME, GridCells, _traces, healpix_pix2ang_ring, so3_healpix_grid, symsol_grid_log_likelihood)
from pose3d.engine.likelihood import LOG_UNIFORM


@pytest.mark.parametrize("nside", [1, 2, 4, 32])
def test_healpix_matches_healpy(nside):
    hp = pytest.importorskip("healpy")
    theta, phi = healpix_pix2ang_ring(nside)
    ref_theta, ref_phi = hp.pix2ang(nside, np.arange(12 * nside * nside))
    np.testing.assert_allclose(theta, ref_theta, atol=1e-12)
    np.testing.assert_allclose(phi, ref_phi, atol=1e-12)


def test_grid_is_image2spheres():
    # Sizes from image2sphere's so3_healpix_grid table; level 5 is the one its log-likelihood uses.
    assert so3_healpix_grid(1).shape == (576, 3, 3)
    assert so3_healpix_grid(2).shape == (4608, 3, 3)
    G = so3_healpix_grid(2).double()
    torch.testing.assert_close(G @ G.transpose(-1, -2), torch.eye(3, dtype=G.dtype).expand_as(G), atol=1e-6, rtol=0)


def test_hits_agree_with_brute_force_nearest():
    torch.manual_seed(0)
    cells = GridCells(rec_level=2)
    labels = matrix_flow.random_rotation_matrices(6)
    # Samples scattered around the labels, so many land in and near the label cells.
    noise = matrix_flow.exp_map(0.15 * torch.randn(6, 500, 3))
    samples = labels[:, None] @ noise
    label_cells = cells.label_cells(labels).view(6, 1, -1, 3, 3)
    hits = cells.hits(samples, label_cells)[:, 0]
    g = _traces(labels, cells.grid).argmax(1)
    nearest = _traces(samples.reshape(-1, 3, 3), cells.grid).argmax(1).view(6, 500)
    torch.testing.assert_close(hits, (nearest == g[:, None]).sum(1))
    assert int(hits.sum()) > 100


class _Data:
    """A stand-in SYMSOL test set: blank images, each with `n_eq` random equivalent poses."""

    def __init__(self, n_img=4, n_eq=3):
        torch.manual_seed(1)
        self.cls = torch.zeros(n_img, dtype=torch.long)
        self.rots = matrix_flow.random_rotation_matrices(n_img * n_eq).view(n_img, n_eq, 3, 3)

    def __getitem__(self, i):
        return dict(img=torch.zeros(3, 4, 4), rots_eq=self.rots[i], n_eq=torch.tensor(self.rots.shape[1]))


class _Uniform(nn.Module):
    """v = 0: the flow leaves its uniform start alone."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def condition(self, img):
        return img.new_zeros(img.shape[0], 1)

    def velocity(self, R, t, cond):
        return 0 * R.reshape(R.shape[0], 9)[:, :3]   # zero, but a function of R for the divergence


class _PointMass(_Uniform):
    """Every sample lands on the image's first equivalent pose."""

    def __init__(self, data):
        super().__init__()
        self.data = data

    def condition(self, img):
        return torch.arange(img.shape[0], dtype=torch.float32)[:, None]   # the images come in order

    def velocity(self, R, t, cond):
        target = self.data.rots[cond[:, 0].long(), 0]
        return matrix_flow.relative_log(R, target) * (t[:, None] == 0)   # one step of dt = 1 lands on it


def test_uniform_flow_scores_the_uniform_density():
    # By counting where cells get many samples (level 1: 576 cells, 100 samples each on average) ...
    data = _Data(n_img=4)
    ll, cls, rare = symsol_grid_log_likelihood(_Uniform(), data, "cpu", {0: 576 * 100}, rec_level=1,
                                               n_equivalents=3, steps=2, log=lambda s: None)
    assert rare == 0
    assert abs(float(ll.mean()) - LOG_UNIFORM) < 0.05
    # ... and by the density where they get none (level 3: 36,864 cells, 0.03 samples each).
    ll, cls, rare = symsol_grid_log_likelihood(_Uniform(), data, "cpu", {0: 1024}, rec_level=3,
                                               n_equivalents=3, steps=2, ll_steps=2, log=lambda s: None)
    assert rare == 4 * 3
    np.testing.assert_allclose(ll, LOG_UNIFORM, atol=1e-5)


def test_point_mass_reaches_the_grid_cap():
    # All mass on one equivalent of three: log(N / pi^2) on its cell. The other two cells get no samples and
    # are scored by the density there, which the backward pass (t > 0, v = 0) sees as uniform.
    data = _Data(n_img=2)
    ll, _, rare = symsol_grid_log_likelihood(_PointMass(data), data, "cpu", {0: 64}, rec_level=1,
                                             n_equivalents=3, steps=1, ll_steps=2, log=lambda s: None)
    cap = math.log(576) - LOG_VOLUME
    assert rare == 2 * 2
    np.testing.assert_allclose(ll, (cap + 2 * LOG_UNIFORM) / 3, atol=1e-5)
