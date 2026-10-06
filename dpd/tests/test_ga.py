"""The geometric algebra and the equivariance claim the Clifford model rests on."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpd.clifford_model import CliffordPAModel
from dpd.data import BL_DEFAULT
from dpd.ga import (
    CliffordAlgebra,
    MVBlock,
    MVGeometricProduct,
    MVGradeGate,
    MVInvariantGate,
    MVLayerNorm,
    MVLinear,
)


@pytest.fixture
def cl20():
    return CliffordAlgebra(2, 0)


def test_cayley_table_of_cl20(cl20):
    e = torch.eye(4, dtype=torch.float64)
    gp = cl20.geometric_product
    np.testing.assert_allclose(gp(e[1], e[1]).numpy(), [1, 0, 0, 0])  # e1 e1 = 1
    np.testing.assert_allclose(gp(e[2], e[2]).numpy(), [1, 0, 0, 0])  # e2 e2 = 1
    np.testing.assert_allclose(gp(e[1], e[2]).numpy(), [0, 0, 0, 1])  # e1 e2 = e12
    np.testing.assert_allclose(gp(e[2], e[1]).numpy(), [0, 0, 0, -1])  # anticommute
    np.testing.assert_allclose(gp(e[3], e[3]).numpy(), [-1, 0, 0, 0])  # e12^2 = -1


def test_signatures_carry_the_right_metric():
    e = torch.eye(4, dtype=torch.float64)
    # Cl(0,2) = quaternions: e1^2 = e2^2 = -1
    q = CliffordAlgebra(0, 2)
    np.testing.assert_allclose(q.geometric_product(e[1], e[1]).numpy(), [-1, 0, 0, 0])
    # a degenerate direction squares to zero
    d = CliffordAlgebra(1, 0, 1)
    np.testing.assert_allclose(d.geometric_product(e[2], e[2]).numpy(), [0, 0, 0, 0])


def test_associativity(cl20):
    g = torch.Generator().manual_seed(0)
    a, b, c = (torch.randn(5, 4, generator=g, dtype=torch.float64) for _ in range(3))
    gp = cl20.geometric_product
    np.testing.assert_allclose(gp(gp(a, b), c).numpy(), gp(a, gp(b, c)).numpy(), atol=1e-12)


def test_rotor_sandwich_is_a_phase_rotation(cl20):
    """The claim the whole model rests on: e^{j phi} acting on I/Q is a rotor."""
    z = torch.complex(torch.randn(64, dtype=torch.float64), torch.randn(64, dtype=torch.float64))
    phi = torch.tensor(0.7321, dtype=torch.float64)
    rotated = cl20.sandwich(cl20.rotor(phi.expand(64)), cl20.embed_complex(z))
    np.testing.assert_allclose(cl20.extract_complex(rotated).numpy(), (z * torch.exp(1j * phi)).numpy(), atol=1e-12)


def test_even_part_is_rotation_invariant(cl20):
    """Blades 0 and e12 must not move under the rotor -- that is what makes them a gain."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(32, 4, generator=g, dtype=torch.float64)
    phi = torch.rand(32, generator=g, dtype=torch.float64) * 6.0
    out = cl20.sandwich(cl20.rotor(phi), a)
    np.testing.assert_allclose(out[:, [0, 3]].numpy(), a[:, [0, 3]].numpy(), atol=1e-12)


def test_grade_sqnorm_is_invariant(cl20):
    g = torch.Generator().manual_seed(2)
    a = torch.randn(16, 4, generator=g, dtype=torch.float64)
    phi = torch.rand(16, generator=g, dtype=torch.float64) * 6.0
    rot = cl20.sandwich(cl20.rotor(phi), a)
    np.testing.assert_allclose(cl20.grade_sqnorm(rot).numpy(), cl20.grade_sqnorm(a).numpy(), atol=1e-12)


def test_reverse_signs(cl20):
    a = torch.arange(4, dtype=torch.float64)
    np.testing.assert_allclose(cl20.reverse(a).numpy(), [0, 1, 2, -3])  # grade 2 flips


@pytest.mark.parametrize(
    "layer_cls", [MVLinear, MVGeometricProduct, MVGradeGate, MVInvariantGate, MVLayerNorm, MVBlock]
)
def test_layers_are_equivariant(cl20, layer_cls):
    """Every layer must commute with the rotor action, or the model's symmetry claim is empty."""
    torch.manual_seed(0)
    layer = MVLinear(cl20, 6, 5) if layer_cls is MVLinear else layer_cls(cl20, 6)
    # perturb the zero-initialised parameters so the test cannot pass trivially
    for param in layer.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.4)

    x = torch.randn(40, 6, 4, dtype=torch.float64)
    phi = torch.tensor(1.234, dtype=torch.float64)
    rotor = cl20.rotor(phi.expand(40, 6))

    rotate_then_apply = layer(cl20.sandwich(rotor, x))
    apply_then_rotate = cl20.sandwich(cl20.rotor(phi.expand(*layer(x).shape[:-1])), layer(x))
    scale = apply_then_rotate.abs().max()
    assert (rotate_then_apply - apply_then_rotate).abs().max() / scale < 1e-10


def test_mvlinear_bias_only_touches_the_scalar_blade(cl20):
    """A bias on any other grade would silently break equivariance."""
    layer = MVLinear(cl20, 3, 2)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([1.5, -2.0]))
    with torch.no_grad():
        out = layer(torch.randn(7, 3, 4, dtype=torch.float64))
    np.testing.assert_allclose(out[..., 1:].numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out[..., 0].numpy(), np.tile([1.5, -2.0], (7, 1)), atol=1e-12)


def test_clifford_pa_model_is_phase_equivariant():
    """Rotating every carrier must rotate the modelled distortion by the same angle."""
    torch.manual_seed(0)
    model = CliffordPAModel(2, bl=BL_DEFAULT, channels=8, n_blocks=2).double()
    for param in model.readout.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.3)

    x = torch.complex(torch.randn(2, 2048, dtype=torch.float64), torch.randn(2, 2048, dtype=torch.float64)) * 0.25
    phi = torch.tensor(2.1, dtype=torch.float64)
    with torch.no_grad():
        y, y_rot = model(x), model(x * torch.exp(1j * phi))
    assert (y_rot - y * torch.exp(1j * phi)).abs().max() / y.abs().max() < 1e-10


def test_clifford_gains_are_phase_invariant():
    """The gains themselves must not move -- they are the AM/AM + AM/PM characteristic."""
    torch.manual_seed(1)
    model = CliffordPAModel(2, bl=BL_DEFAULT, channels=8, n_blocks=1)
    for param in model.readout.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.3)
    x = torch.complex(torch.randn(2, 512, dtype=torch.float64), torch.randn(2, 512, dtype=torch.float64)) * 0.25
    with torch.no_grad():
        phi = torch.tensor(0.9, dtype=torch.float64)
        g, g_rot = model.gains(x), model.gains(x * torch.exp(1j * phi))
    assert (g - g_rot).abs().max() / g.abs().max() < 1e-10


def test_zero_initialised_readout_starts_at_the_no_model_solution():
    model = CliffordPAModel(2, bl=BL_DEFAULT)
    x = torch.complex(torch.randn(2, 256), torch.randn(2, 256)) * 0.2
    with torch.no_grad():
        assert model(x).abs().max() == 0.0


def test_model_rejects_wrong_band_count():
    model = CliffordPAModel(3, bl=BL_DEFAULT)
    with pytest.raises(ValueError, match="expected x of shape"):
        model(torch.complex(torch.randn(2, 128), torch.randn(2, 128)))
