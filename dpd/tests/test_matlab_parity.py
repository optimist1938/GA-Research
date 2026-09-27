"""The port must agree with the MATLAB reference, not merely resemble it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dpd.data import BL_DEFAULT
from dpd.features import ChebyshevLUTFeatures, PartModel
from dpd.matlab import chebyshev_func, fir1, gen_spl_chebyshev, lin2tensor, nmse_db, tensor_order_table
from dpd.models import MemoryPolynomialPA, chebyshev_features


def test_chebyshev_matches_closed_form():
    x = np.linspace(-1, 1, 257)
    for n in range(8):
        np.testing.assert_allclose(chebyshev_func(x, n, 1), np.cos(n * np.arccos(x)), atol=1e-9)


def test_chebyshev_second_kind_seed():
    x = np.linspace(-0.9, 0.9, 33)
    np.testing.assert_allclose(chebyshev_func(x, 1, 2), 2 * x)
    with pytest.raises(ValueError):
        chebyshev_func(x, 3, order=7)


def test_gen_spl_shape_and_endpoints():
    """gen_spl(2^15, ., 1, 6) yields n_basis rows over the magnitude grid."""
    tab = gen_spl_chebyshev(9, 2**15)
    assert tab.shape == (9, 2**15)
    np.testing.assert_allclose(tab[0], 1.0)
    # row k is T_k on 2*((m + 0.5)/2^15 - 0.5)
    t = 2 * ((np.arange(2**15) + 0.5) / 2**15 - 0.5)
    np.testing.assert_allclose(tab[3], chebyshev_func(t, 3, 1), atol=1e-12)


def test_lin2tensor_is_column_major_and_one_based():
    assert lin2tensor([9, 9], 1) == (1, 1)
    assert lin2tensor([9, 9], 2) == (2, 1)
    assert lin2tensor([9, 9], 10) == (1, 2)
    table = tensor_order_table([9, 9])
    assert table.shape == (81, 2)
    for i in range(1, 82):
        assert tuple(table[i - 1] + 1) == lin2tensor([9, 9], i)


def test_fir1_length_and_unit_dc_gain():
    bl = fir1(32, 0.72, "low")
    assert bl.size == 33
    assert bl.sum() == pytest.approx(1.0, abs=1e-9)  # MATLAB scales the passband centre to 1
    with pytest.raises(NotImplementedError):
        fir1(32, 0.72, "high")


def test_nmse_db_definition():
    ref = np.array([1 + 1j, 1 - 1j])
    assert nmse_db(ref, ref) == pytest.approx(0.0)
    assert nmse_db(ref, ref / 10) == pytest.approx(-20.0)


def test_torch_chebyshev_matches_numpy():
    t = torch.linspace(-1, 1, 65, dtype=torch.float64)
    got = chebyshev_features(t, 9).numpy()
    want = np.stack([chebyshev_func(t.numpy(), k, 1) for k in range(9)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def _toy_signals(seed=0, n=4096, dim=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(dim, n)) + 1j * rng.normal(size=(dim, n))
    x /= np.abs(x).max(axis=1, keepdims=True)
    return x


def test_summed_then_filtered_equals_reference_column_form():
    """The torch forward reorders BL out of the per-column loop; it must not change the result."""
    x = _toy_signals(dim=2)
    pm = PartModel.default(2)
    model = MemoryPolynomialPA(pm, [3, 3], BL_DEFAULT, quantise=True)
    rng = np.random.default_rng(1)
    flat = rng.normal(size=model.n_coef) + 1j * rng.normal(size=model.n_coef)
    model.load_flat_coef(flat)

    feats = ChebyshevLUTFeatures(x, pm, [3, 3], BL_DEFAULT)
    _, y_columns = feats.predict(flat)  # convolve every column, then weight and sum
    with torch.no_grad():
        y_torch = model(torch.as_tensor(x)).numpy()  # weight and sum, then convolve once
    np.testing.assert_allclose(y_torch, y_columns, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dim,n_basis", [(1, [3]), (2, [3, 2]), (3, [3, 2, 4])])
def test_torch_forward_matches_regressor_path_in_every_dimension(dim, n_basis):
    """Guards the tensor contraction in the forward pass.

    Contracting the coefficient tensor one basis axis at a time only lines up
    under plain broadcasting when D <= 2; at D = 3 it silently contracts the
    wrong axis.  This check runs each supported dimension against the explicit
    column-by-column regressor construction.
    """
    x = _toy_signals(dim=dim, n=3000)
    pm = PartModel.default(dim)
    model = MemoryPolynomialPA(pm, n_basis, BL_DEFAULT, quantise=True)
    rng = np.random.default_rng(11)
    flat = rng.normal(size=model.n_coef) + 1j * rng.normal(size=model.n_coef)
    model.load_flat_coef(flat)

    feats = ChebyshevLUTFeatures(x, pm, n_basis, BL_DEFAULT)
    _, y_columns = feats.predict(flat)
    with torch.no_grad():
        y_torch = model(torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(y_torch, y_columns, atol=1e-10, rtol=0)


def test_flat_coefficient_round_trip_uses_matlab_ordering():
    pm = PartModel.default(2)
    model = MemoryPolynomialPA(pm, [3, 4], BL_DEFAULT)
    rng = np.random.default_rng(2)
    flat = rng.normal(size=model.n_coef) + 1j * rng.normal(size=model.n_coef)
    model.load_flat_coef(flat)
    np.testing.assert_allclose(model.flat_coef(), flat)
    # column j of part m corresponds to lin2tensor index j + 1
    c = model.coef.detach().numpy()
    for j, order in enumerate(tensor_order_table([4, 5])):
        assert c[0][tuple(order)] == flat[j]


def test_quantised_and_continuous_basis_agree_closely():
    x = _toy_signals(dim=2, n=8192)
    pm = PartModel.default(2)
    rng = np.random.default_rng(3)
    flat = rng.normal(size=(4 * 4) * pm.n_parts) + 1j * rng.normal(size=(4 * 4) * pm.n_parts)
    ys = []
    for quantise in (True, False):
        m = MemoryPolynomialPA(pm, [3, 3], BL_DEFAULT, quantise=quantise)
        m.load_flat_coef(flat)
        with torch.no_grad():
            ys.append(m(torch.as_tensor(x)).numpy())
    # The LUT quantises |x| to a step of 2^-15, which high-order Chebyshev terms
    # amplify by roughly their degree.  The right unit for "close enough" is the
    # error power the model is measured in: on GeoData_TB the two agree to
    # 7e-5 dB of NMSE, so require the difference to sit far below the signal.
    assert nmse_db(ys[0], ys[0] - ys[1]) < -60


def test_least_squares_recovers_planted_coefficients():
    """On data generated by the model itself, LS must recover the output exactly."""
    x = _toy_signals(dim=1, n=8192)
    pm = PartModel.default(1)
    truth = MemoryPolynomialPA(pm, [3], BL_DEFAULT, quantise=True)
    rng = np.random.default_rng(4)
    truth.load_flat_coef(rng.normal(size=truth.n_coef) + 1j * rng.normal(size=truth.n_coef))
    with torch.no_grad():
        d = truth(torch.as_tensor(x)).numpy()

    fitted = MemoryPolynomialPA(pm, [3], BL_DEFAULT, quantise=True)
    fitted.fit_least_squares(x, d)
    with torch.no_grad():
        y = fitted(torch.as_tensor(x)).numpy()
    assert nmse_db(x[0], d - y) < -100


def test_features_reject_unnormalised_input():
    x = _toy_signals(dim=2) * 3.0
    with pytest.raises(ValueError, match="unit peak"):
        ChebyshevLUTFeatures(x, PartModel.default(2), [3, 3], BL_DEFAULT)


def test_features_reject_even_length_filter():
    with pytest.raises(ValueError, match="odd"):
        ChebyshevLUTFeatures(_toy_signals(dim=2), PartModel.default(2), [3, 3], np.ones(4))


def test_part_model_dim_mismatch_is_caught():
    with pytest.raises(ValueError, match="dim 3"):
        ChebyshevLUTFeatures(_toy_signals(dim=2), PartModel.default(3), [3, 3, 3], BL_DEFAULT)


def test_block_streaming_matches_single_block():
    """Chunked Gram accumulation must equal the unchunked matrix, halo included."""
    x = _toy_signals(dim=2, n=5000)
    feats = ChebyshevLUTFeatures(x, PartModel.default(2), [2, 2], BL_DEFAULT)
    _, u_small = feats.matrix(block_size=997)
    _, u_big = feats.matrix(block_size=5000)
    np.testing.assert_allclose(u_small, u_big, atol=1e-12)
