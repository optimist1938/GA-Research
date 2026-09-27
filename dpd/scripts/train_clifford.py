#!/usr/bin/env python3
"""Train the Clifford-equivariant PA model and compare it with the reference.

    python scripts/train_clifford.py --bands AB --steps 3000
    python scripts/train_clifford.py --bands AB --baseline   # LS reference on the same split

Both models are fitted on the same 80% of the record and scored on the held-out
tail, so the comparison is like for like -- the MATLAB script itself reports an
in-sample figure.
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

from dpd.clifford_model import CliffordPAModel  # noqa: E402
from dpd.data import BL_DEFAULT, Split, load_geodata_tb  # noqa: E402
from dpd.features import PartModel  # noqa: E402
from dpd.matlab import nmse_db  # noqa: E402
from dpd.models import MemoryPolynomialPA  # noqa: E402
from dpd.training import Logger, TrainConfig, WandbLogger, evaluate, resolve_device, train  # noqa: E402

from dpd.config import GEODATA as DATA  # noqa: E402  -- the reference recipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, default=DATA)
    ap.add_argument("--bands", default="AB", help="envelopes; the first is the modelled carrier")
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--env-taps", default="-2:9", help="start:stop of the envelope tap range")
    ap.add_argument("--linear-taps", default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="also fit the LS reference on the same split")
    ap.add_argument("--n-basis", default="8,8", help="reference model ModelBasisFuncNum")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="dpd")
    ap.add_argument("--wandb-entity", default="clifforders")
    args = ap.parse_args()

    device = resolve_device(args.device)
    geo = load_geodata_tb(args.data)
    band = args.bands[0]
    ref = geo.bands[band]
    x_np, _ = geo.normalised_stack(args.bands)
    split = Split.contiguous(ref.n)

    x = torch.as_tensor(x_np, dtype=torch.complex64).to(device)
    d = torch.as_tensor(ref.d, dtype=torch.complex64).to(device)

    print(f"band {band} <- {args.bands} on {device}; {ref.n} samples, train {split.train}, val {split.val}")
    print(f"  no-model floor {ref.nmse_no_model():.3f} dB, vendor reference {ref.nmse_reference():.3f} dB\n")

    results = {"band": band, "bands": args.bands, "no_model_db": ref.nmse_no_model(), "reference_db": ref.nmse_reference()}

    if args.baseline:
        nb = [int(v) for v in args.n_basis.split(",")]
        base = MemoryPolynomialPA(PartModel.default(len(args.bands)), nb, BL_DEFAULT, quantise=True)
        t0 = time.time()
        info = base.fit_least_squares(x_np, ref.d, indices=split.train)
        with torch.no_grad():
            y = base(torch.as_tensor(x_np)).numpy()
        results["baseline"] = {
            "n_coef": info["n_coef"],
            "n_basis": nb,
            "fit_seconds": round(time.time() - t0, 1),
            "train_db": nmse_db(ref.x[split.train], (ref.d - y)[split.train]),
            "val_db": nmse_db(ref.x[split.val], (ref.d - y)[split.val]),
        }
        b = results["baseline"]
        print(f"[reference LS] {b['n_coef']} coef  train {b['train_db']:.3f} dB  val {b['val_db']:.3f} dB  ({b['fit_seconds']}s)\n")

    lo, hi = (int(v) for v in args.env_taps.split(":"))
    model = CliffordPAModel(
        n_bands=len(args.bands),
        env_taps=range(lo, hi),
        linear_taps=[int(v) for v in args.linear_taps.split(",")],
        channels=args.channels,
        n_blocks=args.blocks,
        bl=BL_DEFAULT,
    ).to(torch.float32).to(device)

    print(f"[clifford] {model.n_parameters} parameters, {len(model.env_taps)} envelope taps x {len(args.bands)} bands")
    cfg = TrainConfig(steps=args.steps, window=args.window, lr=args.lr, seed=args.seed)
    logger = (
        WandbLogger(project=args.wandb_project, entity=args.wandb_entity, config={**vars(args), "device": str(device)})
        if args.wandb
        else Logger()
    )
    t0 = time.time()
    history = train(model, x, d, ref.x, split, cfg, logger=logger)
    results["clifford"] = {
        "n_parameters": model.n_parameters,
        "channels": args.channels,
        "blocks": args.blocks,
        "env_taps": [lo, hi],
        "linear_taps": model.linear_taps,
        "steps": args.steps,
        "train_seconds": round(time.time() - t0, 1),
        "train_db": evaluate(model, x, d, ref.x, split.train),
        "val_db": evaluate(model, x, d, ref.x, split.val),
        "history": history,
    }
    c = results["clifford"]
    print(f"\n[clifford] train {c['train_db']:.3f} dB  val {c['val_db']:.3f} dB  ({c['train_seconds']}s)")
    logger.finish({"train_db": c["train_db"], "val_db": c["val_db"], "n_parameters": c["n_parameters"]})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
