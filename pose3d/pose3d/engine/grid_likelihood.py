"""SYMSOL log-likelihood on Image2Sphere's grid, so a flow is scored exactly like IPDF / I2S / Howell et al.

Those models output a distribution over a fixed grid of rotations (HEALPix on SO(3), level 5: 2,359,296
points, ~1.9 degrees apart). I2S scores a label R by its nearest grid point g (image2sphere/src/train.py,
evaluate_ll):

    log p(R) = log(P(cell g) * N / pi^2),

the density of a histogram whose cells are the grid's Voronoi cells (volume pi^2 / N each). It is capped at
log(N / pi^2) - log(#equivalents), and a continuous density is not: engine/likelihood.py's exact
log-likelihood rewards precision finer than a cell, which a grid cannot express. To compare with their
tables, P(cell g) of the flow is estimated here by sampling: draw M poses, count those whose nearest grid
point is g. The estimate's relative error is ~1 / sqrt(hits); `samples_per_class` keeps ~40+ hits per cell
for SYMSOL I's shapes (a fraction ~1 / #equivalents of the samples lands in each equivalent's cell).

Finding the nearest of 2.4M grid points for every sample is avoided: a sample can only be in g's cell if it
lies within the grid's covering radius r of g, and then every grid point nearer to it than g lies within
2r of g, so within 3r of the label. Each label keeps its `candidates` nearest grid points (checked to
reach past 3r) and samples are assigned among those.
"""

import math

import numpy as np
import torch

import pose3d.geometry.matrix as matrix_flow
from pose3d.engine.likelihood import flow_log_likelihood, flow_states

LOG_VOLUME = 2.0 * math.log(math.pi)   # SO(3) has volume pi^2 in IPDF's units


def healpix_pix2ang_ring(nside: int):
    """(theta, phi) of every HEALPix pixel centre in RING order; healpy.pix2ang(nside, arange(npix))."""
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    p = np.arange(npix, dtype=np.int64)
    z = np.empty(npix)
    phi = np.empty(npix)

    north = p < ncap
    iring = ((1 + np.sqrt(1 + 2 * p[north].astype(np.float64))) / 2).astype(np.int64)
    iphi = p[north] + 1 - 2 * iring * (iring - 1)
    z[north] = 1 - iring ** 2 * 4.0 / npix
    phi[north] = (iphi - 0.5) * math.pi / (2 * iring)

    belt = (p >= ncap) & (p < npix - ncap)
    ip = p[belt] - ncap
    iring = ip // (4 * nside) + nside
    iphi = ip % (4 * nside) + 1
    fodd = np.where((iring + nside) % 2 == 1, 1.0, 0.5)
    z[belt] = (2 * nside - iring) * 2.0 / (3 * nside)
    phi[belt] = (iphi - fodd) * math.pi / (2 * nside)

    south = p >= npix - ncap
    ip = npix - p[south]
    iring = ((1 + np.sqrt(2 * ip.astype(np.float64) - 1)) / 2).astype(np.int64)
    iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
    z[south] = -1 + iring ** 2 * 4.0 / npix
    phi[south] = (iphi - 0.5) * math.pi / (2 * iring)
    return np.arccos(np.clip(z, -1, 1)), phi


def so3_healpix_grid(rec_level: int = 5) -> torch.Tensor:
    """Image2Sphere's evaluation grid (so3_utils.so3_healpix_grid + o3.angles_to_matrix), (N, 3, 3) float32."""
    from e3nn import o3

    nside = 2 ** rec_level
    beta, alpha = healpix_pix2ang_ring(nside)
    alpha, beta = torch.from_numpy(alpha), torch.from_numpy(beta)
    gamma = torch.linspace(0, 2 * np.pi, 6 * nside + 1)[:-1]
    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, len(beta) // len(gamma))
    abc = torch.stack((alpha, beta, gamma)).float()
    return o3.angles_to_matrix(*abc)


def _traces(A, G):
    """tr(A_i G_j^T) for A (n, 3, 3), G (N, 3, 3): (n, N); larger means nearer (cos = (tr - 1) / 2)."""
    return A.reshape(-1, 9) @ G.reshape(-1, 9).T


def covering_radius(grid: torch.Tensor, n: int = 20000, chunk: int = 128, seed: int = 0) -> float:
    """Largest distance (radians) from a random rotation to its nearest grid point, over n draws."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    R = _quat_to_matrix(q).to(grid)
    best = torch.cat([_traces(R[i:i + chunk], grid).max(1).values for i in range(0, n, chunk)])
    return float(torch.arccos(((best - 1) / 2).clamp(-1, 1)).max())


def _quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


class GridCells:
    """The grid's Voronoi cells, with the covering radius that bounds them."""

    def __init__(self, rec_level: int = 5, device="cpu", candidates: int = 256, radius_margin: float = 1.25):
        self.grid = so3_healpix_grid(rec_level).to(device)
        self.n = self.grid.shape[0]
        self.candidates = candidates
        # Measured on random rotations, with a margin for the worst cell the draws missed.
        self.radius = radius_margin * covering_radius(self.grid)
        self.min_cos_cell = math.cos(self.radius)
        self.max_cos_reach = math.cos(3 * self.radius)

    def label_cells(self, labels: torch.Tensor, chunk: int = 64):
        """For labels (n, 3, 3): the `candidates` nearest grid points (n, K, 3, 3), the first being the label's cell."""
        out = []
        for i in range(0, labels.shape[0], chunk):
            tr = _traces(labels[i:i + chunk].to(self.grid), self.grid)
            top = tr.topk(self.candidates, dim=1)
            # The farthest candidate must be beyond 3 r, or a nearer grid point could have been left out.
            assert float(((top.values[:, -1] - 1) / 2).max()) < self.max_cos_reach, \
                "too few candidate grid points for this grid's covering radius"
            out.append(self.grid[top.indices])
        return torch.cat(out)

    def hits(self, samples: torch.Tensor, cells: torch.Tensor):
        """How many of each image's samples (b, M, 3, 3) fall in each of its label cells (b, k, K, 3, 3): (b, k)."""
        b, k, K = cells.shape[:3]
        tr = torch.einsum("bmij,bkcij->bkmc", samples, cells)               # (b, k, M, K)
        nearest_is_cell = tr.argmax(-1) == 0
        close_enough = (tr[..., 0] - 1) / 2 >= self.min_cos_cell            # within the covering radius of g
        return (nearest_is_cell & close_enough).sum(-1)


@torch.no_grad()
def sample_flow(model, cond, n: int, steps: int = 20):
    """n poses (B * n, 3, 3) per condition row, by the model's Euler sampler from the uniform distribution."""
    states = flow_states(model)
    cond = cond.repeat_interleave(n, dim=0)
    device = cond.device if torch.is_tensor(cond) else next(model.parameters()).device
    s = states.from_matrix(matrix_flow.random_rotation_matrices(cond.shape[0]).to(device))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device)
        s = states.step(s, dt * model.velocity(s, t, cond))
    return states.to_matrix(s)


@torch.no_grad()
def symsol_grid_log_likelihood(model, dataset, device, samples_per_class, rec_level: int = 5,
                               n_equivalents: int = 8, steps: int = 20, min_hits: int = 10, ll_steps: int = 50,
                               chunk_samples: int = 65536, seed: int = 0, cells: GridCells = None, log=print):
    """Image2Sphere's grid log-likelihood of a flow on a SYMSOL test set (datasets/symsol.py, train=False).

    Every image is scored at a random subset of `n_equivalents` of its equivalent poses (I2S uses one random
    equivalent; the mean over several has the same expectation, less noise). `samples_per_class` maps the
    class index to M.

    A cell with fewer than `min_hits` samples is too rare for the count (a diffuse model puts ~M / N in a
    cell: none at all). There P(cell) is the exact density at the label times the cell volume
    (engine/likelihood.py), right where the density is smooth across a cell, capped by the count's ~95 %
    upper bound (hits + 3) / M so a sharp peak just outside the cell cannot score as inside it.
    Returns per-image log-likelihoods, class indices and the number of labels scored that way.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cells = GridCells(rec_level, device) if cells is None else cells
    log(f"grid: {cells.n} rotations, covering radius {math.degrees(cells.radius):.2f} deg (with margin)")
    to_ll = math.log(cells.n) - LOG_VOLUME                                  # log(P(cell) N / pi^2) = log P + to_ll
    model.eval()
    lls, classes, n_rare = [], [], 0
    for c in sorted(samples_per_class):
        idx = (dataset.cls == c).nonzero().flatten().tolist()
        m = samples_per_class[c]
        per_batch = max(1, chunk_samples // m)
        cls_lls, cls_rare = [], 0
        for start in range(0, len(idx), per_batch):
            items = [dataset[i] for i in idx[start:start + per_batch]]
            img = torch.stack([it["img"] for it in items]).to(device)
            labels = []
            for it in items:
                n = int(it["n_eq"])
                pick = torch.randperm(n, generator=gen)[:n_equivalents]
                pick = pick.repeat(math.ceil(n_equivalents / len(pick)))[:n_equivalents]
                labels.append(it["rots_eq"][pick])
            labels = torch.stack(labels).to(device)                                     # (b, k, 3, 3)
            b, k = labels.shape[:2]
            label_cells = cells.label_cells(labels.reshape(-1, 3, 3)).view(b, k, -1, 3, 3)
            cond = model.condition(img)
            samples = sample_flow(model, cond, m, steps).view(b, m, 3, 3).float()
            hits = cells.hits(samples, label_cells).float()
            ll = torch.log(hits.clamp(min=1e-9) / m) + to_ll                             # (b, k)
            rare = hits < min_hits
            if rare.any():
                rows = rare.nonzero()
                exact = flow_log_likelihood(model, cond[rows[:, 0]], labels[rows[:, 0], rows[:, 1]], steps=ll_steps)
                bound = torch.log((hits[rare] + 3) / m) + to_ll
                ll[rare] = torch.minimum(exact.to(ll), bound)
                cls_rare += int(rare.sum())
            cls_lls.append(ll.mean(1).cpu())
        cls_lls = torch.cat(cls_lls)
        log(f"class {c}: {len(idx)} images, {m} samples each, grid log-likelihood {float(cls_lls.mean()):.3f}"
            f" ({cls_rare} of {len(idx) * n_equivalents} labels under {min_hits} hits, scored by density)")
        lls.append(cls_lls)
        classes.append(torch.full((len(idx),), c))
        n_rare += cls_rare
    return torch.cat(lls).numpy(), torch.cat(classes).numpy(), n_rare
