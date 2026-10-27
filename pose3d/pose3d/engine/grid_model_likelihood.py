"""SYMSOL log-likelihood of a model that outputs a distribution over Image2Sphere's SO(3) grid (I2SReal).

This is upstream Image2Sphere's `evaluate_ll` (image2sphere/train.py), applied to every equivalent pose the
loader hands over instead of the one random label upstream draws:

    log p(R) = log(softmax(grid_logits)[g] * N / pi^2),    g = the grid rotation nearest to R,

the density of a histogram whose cells are the grid's Voronoi cells (volume pi^2 / N each). As upstream, it
is not capped. The flows are scored on the same grid by sampling (engine/grid_likelihood.py); their exact
density is in engine/likelihood.py. Units follow IPDF: SO(3) has volume pi^2, so the uniform distribution
scores -2.29 and a higher number means more mass on the true poses.
"""

import math

import torch

LOG_VOLUME = 2.0 * math.log(math.pi)


@torch.no_grad()
def nearest_grid_index(labels: torch.Tensor, grid: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """Index of the grid rotation nearest to each label, (n, 3, 3) x (Q, 3, 3) -> (n,). tr(A B^T) is largest for
    the nearest rotation (cos of the angle = (tr - 1) / 2), so no angle is computed."""
    g = grid.reshape(-1, 9).T
    out = [(labels[i:i + chunk].reshape(-1, 9).to(g) @ g).argmax(1) for i in range(0, labels.shape[0], chunk)]
    return torch.cat(out)


@torch.no_grad()
def grid_model_log_likelihood(model, loader, device, n_equivalents: int = 8, seed: int = 0):
    """Per-image mean log-likelihood over `n_equivalents` of the image's equivalent poses (all of them when it
    has fewer), with the loader's `rots_eq` / `n_eq` (datasets/symsol.py, train=False). The equivalents are
    drawn with the same generator and order as engine/likelihood.py's symsol_log_likelihood, so the flows and
    the grid models are scored on the same labels. Returns (log-likelihoods (N,), class indices (N,))."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    _, grid = model._eval_grid(device)
    to_ll = math.log(grid.shape[0]) - LOG_VOLUME   # log(P(cell) * N / pi^2) = log P(cell) + to_ll
    lls, classes = [], []
    for batch in loader:
        img = batch["img"].to(device)
        rots, n_eq = batch["rots_eq"], batch["n_eq"]
        b = img.shape[0]
        idx = torch.stack([torch.randperm(int(n), generator=gen)[:n_equivalents].repeat(
            math.ceil(n_equivalents / min(int(n), n_equivalents)))[:n_equivalents] for n in n_eq])
        picked = rots[torch.arange(b)[:, None], idx].reshape(-1, 3, 3).to(device)        # (b * k, 3, 3)
        weight = torch.stack([(torch.arange(n_equivalents) < min(int(n), n_equivalents)).float()
                              for n in n_eq])                                          # repeats count once
        cell = nearest_grid_index(picked, grid).view(b, n_equivalents)
        logp = torch.log_softmax(model.grid_logits(img), dim=1)                        # (b, Q)
        ll = (torch.gather(logp, 1, cell) + to_ll).cpu()                                # (b, k)
        lls.append((ll * weight).sum(1) / weight.sum(1))
        classes.append(batch["cls_eval"].view(-1))
    return torch.cat(lls).numpy(), torch.cat(classes).numpy()
