"""GATr denoiser: embedding fidelity and CliffordFlow integration (no pretrained weights)."""

import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.models.clifford_flow import CliffordFlow
from pose3d.models.gatr_denoiser import _PGA_INDEX, _ensure_xformers_stub

_ensure_xformers_stub()
pytest.importorskip("gatr", reason="GATr package not installed")

ALGEBRA = CliffordAlgebra((1, 1, 1))


def test_pga_embedding_is_an_algebra_homomorphism():
    _ensure_xformers_stub()
    from gatr.primitives import geometric_product

    a, b = torch.randn(4, 8), torch.randn(4, 8)

    def emb(x):
        out = torch.zeros(4, 16)
        out[:, list(_PGA_INDEX)] = x
        return out

    prod = geometric_product(emb(a), emb(b))
    torch.testing.assert_close(prod[:, list(_PGA_INDEX)], ALGEBRA.geometric_product(a, b),
                               atol=1e-5, rtol=1e-5)


def _flow(vector_field):
    return CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                        vector_field=vector_field,
                        gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2))


def test_vector_field_shape_and_zero_init():
    field = _flow("gatr").vector_field
    out = field(torch.randn(3, 10, 8))
    assert out.shape == (3, 1, 8)
    assert out.abs().max() == 0


def test_flow_trains_and_samples():
    torch.manual_seed(0)
    model = _flow("gatr")
    img = torch.randn(2, 3, 64, 64)
    rot = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    rot = rot * torch.linalg.det(rot).sign().view(-1, 1, 1)
    loss = model.compute_loss(img, rot)
    loss.backward()
    grads = [p.grad for p in model.vector_field.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.predict(img, n_samples=2, steps=3).shape == (2, 3, 3)


def test_default_is_still_the_clifford_mlp():
    assert type(_flow("clifford").vector_field).__name__ == "TralaleroTralala"


def _flow_both():
    return CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                        vector_field="gatr", condition_head="gatr",
                        gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2))


def test_condition_head_shape_and_token_identity():
    from pose3d.models.gatr_denoiser import GATrConditionHead

    head = GATrConditionHead(12, 5, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2)
    x = torch.randn(3, 12, 8)
    out = head(x)
    assert out.shape == (3, 5, 8)
    # tokens carry a learned identity, so permuting them must change the output
    assert not torch.allclose(out, head(x[:, torch.randperm(12)]), atol=1e-4)


def test_flow_with_gatr_condition_head_trains_and_samples():
    torch.manual_seed(0)
    model = _flow_both()
    assert type(model.condition_head).__name__ == "GATrConditionHead"
    img = torch.randn(2, 3, 64, 64)
    rot = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    rot = rot * torch.linalg.det(rot).sign().view(-1, 1, 1)
    model.compute_loss(img, rot).backward()
    for part in (model.condition_head, model.vector_field):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.predict(img, n_samples=2, steps=3).shape == (2, 3, 3)


def test_gatr_condition_head_rejects_incompatible_options():
    with pytest.raises(ValueError):
        CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                     encoder_type="resnet50", conv_adapter=False, condition_head="gatr",
                     mlp_heads=True)
