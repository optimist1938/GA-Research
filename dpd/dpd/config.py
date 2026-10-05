"""The reference recipe for the dpd experiments, and the feature flags.

Per the repository convention, the defaults here *are* the reference recipe:
the configuration that reproduces the best measured score on `main`. A change
that is measured better lands with its default updated; a change that is not yet
proven lands with its flag off, one argument away from being tried.

Current best: **-24.139 dB** held-out NMSE on band A of `GeoData_TB`, from the
Chebyshev memory polynomial over the band A and band B envelopes.
"""

from __future__ import annotations

import dataclasses
import pathlib

__all__ = ["DATA_DIR", "GEODATA", "DOV2", "ReferenceRecipe", "CliffordRecipe", "REFERENCE", "CLIFFORD"]

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
GEODATA = DATA_DIR / "GeoData_TB.mat"
DOV2 = DATA_DIR / "DOV2.mat"


@dataclasses.dataclass
class ReferenceRecipe:
    """The ported MATLAB model -- the score to beat.

    ``bands`` is the set of envelopes feeding the nonlinearity; the first is the
    carrier being modelled. Band C is excluded from the default deliberately:
    it is 813 MHz from band A and worth 0.33 dB out of sample for 2.8x the
    coefficients, which is not a trade the reference recipe makes.
    """

    band: str = "A"
    bands: str = "AB"
    n_basis: tuple[int, ...] = (8, 8)
    # 'matlab_pinv' reproduces the MATLAB solution bit for bit; 'ridge' generalises
    # better at D = 3, where the normal equations are conditioned at ~4e21.
    solver: str = "matlab_pinv"
    ridge_alpha: float = 1e-6
    # Reproduce MATLAB's 2^15-entry magnitude lookup exactly. Off makes the model
    # differentiable w.r.t. the envelope, at a cost of 6.7e-5 dB.
    quantise: bool = True
    abs_scale: float = 0.99
    train_frac: float = 0.8
    guard: int = 512


@dataclasses.dataclass
class CliffordRecipe:
    """The geometric-algebra model. Not yet competitive -- see vibe/clifford-model.md.

    Exactly phase-equivariant by construction, but -12.2 dB against the
    reference's -24.1 dB. The diagnosis is the trunk, not the optimiser: a random
    trunk with a closed-form optimal readout already reaches -11.5 dB.
    """

    band: str = "A"
    bands: str = "AB"
    env_taps: tuple[int, int] = (-2, 9)
    linear_taps: tuple[int, ...] = (0, 1, 2)
    channels: int = 16
    n_blocks: int = 2
    gate: str = "invariant"
    # Normalising across channels divides out the envelope scale, which for a PA
    # is the signal rather than a nuisance. Off by default.
    normalise: bool = False
    steps: int = 3000
    window: int = 8192
    lr: float = 3e-3
    seed: int = 0


REFERENCE = ReferenceRecipe()
CLIFFORD = CliffordRecipe()
