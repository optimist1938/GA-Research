"""Experiment configuration: the single place that decides what a default run is.

The defaults are the reference recipe: Huawei's LUT model (`--model=lut_ls`), a Chebyshev
tensor over the three band magnitudes, solved by least squares.

Experiment workflow (see the root README and the idea board in dpd/README.md): a change that is
proven better becomes the default here; a change not yet proven lands switched off; a change
that turned out worse is written up in a report instead.

Every field below is also a command-line flag of the same name (`--model gated`,
`--no-cross_time_invariants`, ...). Sections only group them; flag names are globally unique.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List, Literal, Optional, get_args, get_origin, get_type_hints

ModelName = Literal[
    "lut_ls",    # Huawei-style x_c(k-m) * LUT(|xA|,|xB|,|xC|), least squares (reference)
    "lut_cp",    # the same LUT factorized as a rank-R CP product of 1D functions
    "mlp_poly",  # the team's first MLP on polynomial features of x and conj(x), not equivariant
    "gated",     # U(1)^3-equivariant: x_c(k-m) * g_m(invariants), g a small MLP
    "gated_tcn", # the same gating, g a dilated 1D conv net over the invariant time series
]


@dataclass
class RunConfig:
    """Where the data is and where results go."""

    data_path: str = "data/GeoData_TB.mat"
    out_dir: str = "runs"
    run_name: Optional[str] = None
    seed: int = 0
    train_frac: float = 0.8   # train = first 80% of the series, test = the last 20%
    psd_plot: bool = True
    # Project the model output onto |f| <= band_limit_mhz (d itself has <2e-4 of its power outside
    # +-120 MHz, like Huawei's BL filter). 0 = off.
    band_limit_mhz: float = 0.0
    save_pred: bool = False   # write the model output (3, N) to runs/<run_name>_pred.npy (for ensembles)


@dataclass
class ModelConfig:
    model: ModelName = "lut_ls"
    # Delays m of the carrier term x_c(k-m): m in -memory..memory.
    memory: int = 3
    # Fit lut_ls first and train the model on its residual d - lut(x); the output is their sum.
    residual_lut: bool = False


@dataclass
class LutConfig:
    """lut_ls / lut_cp: basis functions of the band magnitudes."""

    lut_basis: Literal["chebyshev", "even_poly"] = "chebyshev"
    # chebyshev: T_0..T_order of |x| per band (tensor product);
    # even_poly: |x|^(2n) monomials of total degree <= order.
    lut_order: int = 5
    ridge: float = 1e-9        # Tikhonov term, relative to the mean diagonal of the Gram matrix
    cp_rank: int = 8           # lut_cp only


@dataclass
class NetConfig:
    """mlp_poly / gated: the neural parts."""

    # Taps k-inv_memory..k+inv_memory whose invariants feed the gate network.
    inv_memory: int = 10
    cross_time_invariants: bool = True
    max_lag_diff: int = 2      # Re/Im x_b(k-i) conj(x_b(k-j)) for 1 <= j-i <= max_lag_diff
    hidden: int = 128
    layers: int = 3            # adopted: 3 layers beat 2 by +0.3…+0.7 dB (round 2, 3 seeds)
    poly_memory: int = 20      # mlp_poly: taps k-poly_memory+1..k
    per_band: bool = True      # adopted (round 2): a separate gate MLP per output band, +0.1…+2.7 dB
    # gated_tcn: residual blocks of kernel-3 convs with dilations 1, 2, ..., 2^(tcn_levels-1).
    tcn_levels: int = 5
    tcn_channels: int = 64
    tcn_window: int = 1024     # training sequence length


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    threads: int = 0           # torch CPU threads; 0 = every core


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lut: LutConfig = field(default_factory=LutConfig)
    net: NetConfig = field(default_factory=NetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _add_flag(parser, fld: dataclasses.Field, tp) -> None:
    if get_origin(tp) is not None and type(None) in get_args(tp):   # Optional[X]
        tp = next(a for a in get_args(tp) if a is not type(None))
    name = "--" + fld.name
    if tp is bool:
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=fld.default)
    elif get_origin(tp) is Literal:
        parser.add_argument(name, choices=get_args(tp), default=fld.default)
    else:
        parser.add_argument(name, type=tp, default=fld.default)


def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Fit a 3-band DPD model. Defaults are the reference recipe.")
    sections = get_type_hints(Config)
    for name, cls in sections.items():
        group = parser.add_argument_group(name, (cls.__doc__ or "").strip().splitlines()[0]
                                          if cls.__doc__ and not cls.__doc__.startswith(cls.__name__ + "(")
                                          else None)
        hints = get_type_hints(cls)
        for fld in dataclasses.fields(cls):
            _add_flag(group, fld, hints[fld.name])
    ns = vars(parser.parse_args(argv))
    cfg = Config()
    for name, cls in sections.items():
        setattr(cfg, name, cls(**{f.name: ns[f.name] for f in dataclasses.fields(cls)}))
    return cfg
