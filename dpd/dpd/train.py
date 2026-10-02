"""Fit one model on the train part of the series and report test NMSE per band.

    python -m dpd --model gated --epochs 40
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from dpd.config import Config, parse_args
from dpd.data import BANDS, load, split_index
from dpd.metrics import band_project, nmse_db, psd_plot
from dpd.models import build


def run(cfg: Config) -> dict:
    np.random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    if cfg.train.threads:
        torch.set_num_threads(cfg.train.threads)

    sig = load(cfg.run.data_path)
    train, test = split_index(sig.n, cfg.run.train_frac)
    # One positive scale per band: keeps the phase symmetry and every NMSE unchanged.
    scale = 1 / np.sqrt(np.mean(np.abs(sig.x[:, train]) ** 2, axis=1, keepdims=True))
    x, d, e_ref = sig.x * scale, sig.d * scale, sig.e_ref * scale

    model = build(cfg)
    t0 = time.time()
    model.fit(x, d, train, test)
    fit_s = time.time() - t0
    y = model.predict(x)
    if cfg.run.band_limit_mhz:
        y = band_project(y, sig.fs, cfg.run.band_limit_mhz * 1e6)
    err = d - y

    name = cfg.run.run_name or cfg.model.model
    res = {"name": name, "model": cfg.model.model, "n_params": model.n_params, "fit_seconds": fit_s,
           "config": cfg.to_dict(), "bands": {}}
    print(f"\n{name}: {model.n_params} real parameters, fit {fit_s:.0f}s. Test NMSE:")
    print(f"{'band':>4} {'vs d':>8} {'vs x':>8} {'eRef vs x':>10}")
    for c, b in enumerate(BANDS):
        r = {"nmse_vs_d": nmse_db(err[c, test], d[c, test]),
             "nmse_vs_x": nmse_db(err[c, test], x[c, test]),
             "eref_vs_x": nmse_db(e_ref[c, test], x[c, test])}
        res["bands"][b] = r
        print(f"{b:>4} {r['nmse_vs_d']:8.2f} {r['nmse_vs_x']:8.2f} {r['eref_vs_x']:10.2f}")

    out = pathlib.Path(cfg.run.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=2))
    if cfg.run.save_pred:
        np.save(out / f"{name}_pred.npy", (y / scale).astype(np.complex64))
    if cfg.run.psd_plot:
        for c, b in enumerate(BANDS):
            psd_plot(out / f"{name}_psd_{b}.png", sig.fs,
                     {"x": x[c, test], "d": d[c, test], "error d - f(x)": err[c, test],
                      "eRef": e_ref[c, test]},
                     f"{name}, band {b} (test)")
    return res


def main():
    run(parse_args())
