"""SYMSOL log-likelihood of a model that outputs a distribution over Image2Sphere's SO(3) HEALPix grid (the Wigner-D regressor), scored the way
IPDF / Image2Sphere / Howell et al. and engine-side flow evaluations score theirs:

    log p(R) = log(P(cell g) * N / pi^2),   g = the grid point nearest to the label R,   P = softmax over the grid,

capped at log(N / pi^2) - log(#equivalents) (what a model that spreads its mass evenly over the equivalents' cells can reach). Every test view is
scored at `n_equivalents` of its equivalent poses, drawn with the same generator and order as the flow evaluation (seed 0, class by class), so both
models see the same labels. The average over the shapes' means is the number in Howell et al.'s tables.

A regressor's scores <Psi, D(R)> are not trained as logits, so the softmax needs a temperature: T = 1 is the paper-faithful reading; calibrated_temperature
picks the best T of a grid on a calibration subset of the test views (the first `n_cal` of each shape) and reports on the rest.
"""

import math

import numpy as np
import torch

LOG_VOLUME = 2.0 * math.log(math.pi)          # SO(3) has volume pi^2 in IPDF's units
TEMPERATURES = tuple(2.0 ** k for k in range(0, 10))


@torch.no_grad()
def nearest_cell(labels: torch.Tensor, grid: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """Index of the grid rotation nearest to each label: (n, 3, 3) x (Q, 3, 3) -> (n,). Larger trace = nearer."""
    g = grid.reshape(-1, 9).T
    out = [(labels[i:i + chunk].reshape(-1, 9).to(grid) @ g).argmax(1) for i in range(0, labels.shape[0], chunk)]
    return torch.cat(out)


@torch.no_grad()
def wigner_grid_log_likelihood(model, dataset, device, n_equivalents: int = 8, temperatures=TEMPERATURES, batch: int = 16, seed: int = 0,
                               log=print):
    """Per-image grid log-likelihood of `model` (needs .grid_logits(img) and ._eval_grid(device)) on a SYMSOL test set.

    Returns (ll of shape (len(temperatures), N), class index (N,), the cap per image (N,)), images in class-major order.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    _, rots = model._eval_grid(device)
    to_ll = math.log(rots.shape[0]) - LOG_VOLUME
    temps = torch.tensor(temperatures, device=device).view(-1, 1, 1)
    lls, classes, caps = [], [], []
    for c in sorted(set(dataset.cls.tolist())):
        idx = (dataset.cls == c).nonzero().flatten().tolist()
        cls_ll = []
        for start in range(0, len(idx), batch):
            items = [dataset[i] for i in idx[start:start + batch]]
            img = torch.stack([it["img"] for it in items]).to(device)
            labels, n_eq = [], []
            for it in items:
                n = int(it["n_eq"]); n_eq.append(n)
                pick = torch.randperm(n, generator=gen)[:n_equivalents]
                pick = pick.repeat(math.ceil(n_equivalents / len(pick)))[:n_equivalents]
                labels.append(it["rots_eq"][pick])
            labels = torch.stack(labels).to(device)                                          # (b, k, 3, 3)
            b, k = labels.shape[:2]
            cell = nearest_cell(labels.reshape(-1, 3, 3), rots).view(b, k)
            logits = model.grid_logits(img).float()                                          # (b, Q)
            scaled = logits[None] / temps                                                    # (T, b, Q)
            logp = torch.gather(scaled, 2, cell[None].expand(len(temperatures), b, k)) - torch.logsumexp(scaled, dim=2, keepdim=True)
            ll = (logp + to_ll).mean(2)                                                      # (T, b)
            cap = to_ll - torch.log(torch.tensor(n_eq, device=device, dtype=torch.float))
            cls_ll.append(torch.minimum(ll, cap[None]).cpu())
            caps.append(cap.cpu())
        cls_ll = torch.cat(cls_ll, dim=1)
        log(f"class {c}: {len(idx)} images, grid log-likelihood at T=1 {float(cls_ll[0].mean()):.3f}")
        lls.append(cls_ll)
        classes.append(torch.full((len(idx),), c))
    return torch.cat(lls, dim=1).numpy(), torch.cat(classes).numpy(), torch.cat(caps).numpy()


def summarize(ll, classes, temperatures=TEMPERATURES, n_cal: int = 500):
    """Shape means and their average at T = 1, and at the temperature calibrated on each shape's first n_cal views, scored on the remaining views."""
    shapes = sorted(set(classes.tolist()))
    pos = {c: np.nonzero(classes == c)[0] for c in shapes}
    n_cal = min(n_cal, min(len(p) for p in pos.values()) // 2)           # leave views to report on
    cal = np.concatenate([p[:n_cal] for p in pos.values()])
    rest = {c: p[n_cal:] for c, p in pos.items()}
    # best T by the average over shapes of the shape means on the calibration views
    score = [np.mean([ll[t][np.intersect1d(pos[c][:n_cal], cal)].mean() for c in shapes]) for t in range(len(temperatures))]
    best = int(np.argmax(score))
    out = {
        "log_likelihood": float(np.mean([ll[0][pos[c]].mean() for c in shapes])),
        "log_likelihood_calibrated": float(np.mean([ll[best][rest[c]].mean() for c in shapes])),
        "log_likelihood_temperature": float(temperatures[best]),
    }
    for c in shapes:
        out[f"log_likelihood_class{c}"] = float(ll[0][pos[c]].mean())
        out[f"log_likelihood_calibrated_class{c}"] = float(ll[best][rest[c]].mean())
    out["log_likelihood_by_temperature"] = {f"{t:g}": float(np.mean([ll[i][pos[c]].mean() for c in shapes])) for i, t in enumerate(temperatures)}
    return out


def symsol_final_log_likelihood(model, cfg, run=None, log=print):
    """Score a trained grid model on the SYMSOL test set (cfg.data.symsol_ll_views per shape) and log final_* metrics to W&B / the offline sync."""
    from pose3d.datasets.symsol import Symsol
    from pose3d.engine.tracking import log_offline_sync

    d = cfg.data
    test = Symsol(cfg.run.path_to_datasets, train=False, set_number=d.symsol_set, num_views=d.symsol_ll_views)
    log(f"SYMSOL log-likelihood on {len(test)} test views")
    ll, classes, _ = wigner_grid_log_likelihood(model, test, cfg.device, n_equivalents=d.symsol_ll_equivalents, log=log)
    out = summarize(ll, classes)
    metrics = {f"final_{k}": v for k, v in out.items() if k != "log_likelihood_by_temperature"}
    log("grid log-likelihood by temperature: " + "  ".join(f"T={k}: {v:.3f}" for k, v in out["log_likelihood_by_temperature"].items()))
    log(f"FINAL grid log-likelihood (T=1): {out['log_likelihood']:.3f}; calibrated T={out['log_likelihood_temperature']:g}: {out['log_likelihood_calibrated']:.3f}")
    log_offline_sync(run, cfg, step=cfg.train.n_epochs, metrics=metrics, final=True)
    if run is not None:
        run.summary.update(metrics)
    return out
