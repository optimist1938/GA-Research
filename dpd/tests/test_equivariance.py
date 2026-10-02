"""Rotating band b by e^{i a_b} must rotate output band b by the same phase (U(1)^3).

gated and the LUT models hold this by construction; mlp_poly does not (it reads raw Re / Im parts),
which is the point of the comparison.
"""

import numpy as np
import pytest

from dpd.config import Config
from dpd.models import build


def _signals(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))) / np.sqrt(2)
    d = x * np.abs(x) ** 2 + 0.3 * x * np.abs(np.roll(x, 1, axis=0)) ** 2   # a U(1)^3-equivariant target
    return x, d


def _fitted(kind, **overrides):
    cfg = Config()
    cfg.model.model = kind
    cfg.model.memory = 1
    cfg.lut.lut_order = 3
    cfg.lut.cp_rank = 2
    cfg.net.inv_memory, cfg.net.poly_memory, cfg.net.hidden = 2, 3, 16
    cfg.net.tcn_levels, cfg.net.tcn_channels, cfg.net.tcn_window = 2, 8, 256
    cfg.train.epochs = 1
    for section_field, value in overrides.items():
        section, name = section_field.split("__")
        setattr(getattr(cfg, section), name, value)
    x, d = _signals()
    model = build(cfg)
    model.fit(x, d, slice(0, 2000), slice(2000, 3000))
    return model, x


@pytest.mark.parametrize("kind, overrides", [
    ("lut_ls", {}), ("lut_cp", {}), ("gated", {}), ("gated_tcn", {}),
    ("gated", {"net__per_band": True}), ("gated", {"model__residual_lut": True}),
])
def test_equivariant(kind, overrides):
    model, x = _fitted(kind, **overrides)
    phase = np.exp(1j * np.array([0.7, -1.9, 2.8]))[:, None]
    np.testing.assert_allclose(model.predict(x * phase), model.predict(x) * phase, atol=1e-4)


def test_mlp_poly_is_not_equivariant():
    model, x = _fitted("mlp_poly")
    phase = np.exp(1j * np.array([0.7, -1.9, 2.8]))[:, None]
    assert np.abs(model.predict(x * phase) - model.predict(x) * phase).max() > 1e-3
