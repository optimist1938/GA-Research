#!/usr/bin/env python3
"""Reproduce the supplied MATLAB result on GeoData_TB.

Runs the ported Chebyshev memory polynomial over a sweep of nonlinearity
dimensions and band orderings, in-sample (as the MATLAB script does) and
optionally on a held-out split.

    python scripts/reproduce_matlab.py --band A --bands AB --n-basis 8,8
    python scripts/reproduce_matlab.py --sweep
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dpd.data import BL_DEFAULT, Split, load_geodata_tb  # noqa: E402
from dpd.features import PartModel  # noqa: E402
from dpd.metrics import summarise  # noqa: E402
from dpd.models import MemoryPolynomialPA  # noqa: E402
from dpd.solvers import effective_rank  # noqa: E402

from dpd.config import GEODATA as DATA  # noqa: E402  -- the reference recipe


def run_one(geo, band: str, bands: str, n_basis, solver: str, holdout: bool, rank: bool, **solver_kwargs):
    x, _ = geo.normalised_stack(bands)
    ref = geo.bands[band]
    if bands[0] != band:
        raise ValueError(f"band order {bands!r} must start with the modelled band {band!r}")

    pm = PartModel.default(len(bands))
    model = MemoryPolynomialPA(pm, n_basis, BL_DEFAULT, quantise=True)

    t0 = time.time()
    index = Split.contiguous(ref.n).train if holdout else None
    info = model.fit_least_squares(x, ref.d, indices=index, solver=solver, **solver_kwargs)
    fit_s = time.time() - t0

    with torch.no_grad():
        y = model(torch.as_tensor(x)).numpy()

    row = {
        "band": band,
        "bands": bands,
        "dim": len(bands),
        "n_basis": list(n_basis),
        "n_coef": info["n_coef"],
        "solver": solver,
        "fit_seconds": round(fit_s, 1),
        **summarise(ref.x, ref.d, y, ref.e_ref),
    }
    if holdout:
        split = Split.contiguous(ref.n)
        from dpd.matlab import nmse_db

        row["nmse_train_db"] = nmse_db(ref.x[split.train], (ref.d - y)[split.train])
        row["nmse_val_db"] = nmse_db(ref.x[split.val], (ref.d - y)[split.val])
    if rank:
        r, cond = effective_rank(info["rx"])
        row["numerical_rank"] = r
        row["cond_rx"] = f"{cond:.3e}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=DATA)
    ap.add_argument("--band", default="A", choices=list("ABC"), help="carrier to model")
    ap.add_argument("--bands", default="AB", help="envelopes feeding the nonlinearity; must start with --band")
    ap.add_argument("--n-basis", default="8,8", help="ModelBasisFuncNum, comma separated")
    ap.add_argument("--solver", default="matlab_pinv", choices=["matlab_pinv", "ridge"])
    ap.add_argument("--holdout", action="store_true", help="fit on 80%% and report the held-out NMSE")
    ap.add_argument("--rank", action="store_true", help="report the Gram matrix conditioning")
    ap.add_argument("--sweep", action="store_true", help="run the full dimension/band sweep")
    ap.add_argument("--out", type=pathlib.Path, help="write results as JSON")
    args = ap.parse_args()

    geo = load_geodata_tb(args.data)
    print(f"GeoData_TB: {geo.bands['A'].n} samples @ {geo.fs_hz / 1e6:.2f} MHz")
    for b in "ABC":
        bd = geo.bands[b]
        print(
            f"  band {b} (f = {bd.carrier_hz / 1e9:.4f} GHz): "
            f"no-model {bd.nmse_no_model():.3f} dB, vendor reference {bd.nmse_reference():.3f} dB"
        )
    print()

    if args.sweep:
        configs = [
            ("A", "A", [8]),
            ("A", "AB", [8, 8]),
            ("A", "AC", [8, 8]),
            ("A", "ABC", [8, 4, 4]),
            ("B", "BA", [8, 8]),
            ("C", "CA", [8, 8]),
            ("C", "CB", [8, 8]),
        ]
    else:
        configs = [(args.band, args.bands, [int(v) for v in args.n_basis.split(",")])]

    rows = []
    for band, bands, nb in configs:
        print(f"[fit] band {band} <- {bands}, n_basis={nb}", flush=True)
        row = run_one(geo, band, bands, nb, args.solver, args.holdout, args.rank)
        rows.append(row)
        extra = ""
        if args.holdout:
            extra = f"  train {row['nmse_train_db']:.3f} / val {row['nmse_val_db']:.3f} dB"
        if args.rank:
            extra += f"  rank {row['numerical_rank']}/{row['n_coef']} cond {row['cond_rx']}"
        print(f"      NMSE {row['nmse_model_db']:.5f} dB  ({row['n_coef']} coef, {row['fit_seconds']}s){extra}\n", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
