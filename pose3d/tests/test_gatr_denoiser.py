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


def _grads(module, out, inputs):
    """Output, then the gradients of a fixed random projection of it w.r.t. inputs and parameters."""
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=out.dtype)
    module.zero_grad()
    for x in inputs:
        x.grad = None
    (out * weight).sum().backward()
    params = [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in module.parameters()]
    return [out.detach()] + [x.grad.clone() for x in inputs] + params


@pytest.mark.parametrize("in_mv,out_mv,in_s,out_s,with_scalars", [
    (1, 8, 2, 32, True), (8, 8, 32, 32, True), (8, 8, 32, 32, False), (8, 8, None, None, False),
    (8, 16, 32, None, True), (8, 4, 32, None, True), (16, 8, 32, 32, True), (8, 1, 32, None, True),
])
def test_dense_equilinear_matches_gatr(in_mv, out_mv, in_s, out_s, with_scalars):
    import copy
    from gatr.layers.linear import EquiLinear
    from pose3d.models.gatr_denoiser import use_dense_linears

    torch.manual_seed(0)
    ref = EquiLinear(in_mv, out_mv, in_s_channels=in_s, out_s_channels=out_s).double()
    for p in ref.parameters():
        torch.nn.init.normal_(p)   # non-zero biases too
    dense = use_dense_linears(copy.deepcopy(ref))
    assert type(dense) is not EquiLinear and isinstance(dense, EquiLinear)

    mv = torch.randn(3, 5, in_mv, 16, dtype=torch.float64)
    s = torch.randn(3, 5, in_s, dtype=torch.float64) if with_scalars else None
    results = []
    for m in (ref, dense):
        x = mv.clone().requires_grad_()
        sc = None if s is None else s.clone().requires_grad_()
        out_mv_, out_s_ = m(x, scalars=sc)
        out = out_mv_.flatten(-2) if out_s_ is None else torch.cat([out_mv_.flatten(-2), out_s_], -1)
        results.append(_grads(m, out, [x] if sc is None else [x, sc]))
    for a, b in zip(*results):
        torch.testing.assert_close(b, a, atol=1e-12, rtol=1e-12)


def _gatr_pair(cls, args, **flags):
    """A default GATr module and a flagged one with the same weights."""
    torch.manual_seed(0)
    ref = cls(*args, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.3)   # the read-out is zero-initialised otherwise
    other = cls(*args, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2, **flags)
    assert other.state_dict().keys() == ref.state_dict().keys()
    other.load_state_dict(ref.state_dict())
    return ref, other


@pytest.mark.parametrize("head", [False, True])
def test_dense_linear_flag_keeps_weights_and_outputs(head):
    from pose3d.models.gatr_denoiser import GATrConditionHead, GATrVectorField

    cls, args, n_in = (GATrConditionHead, (12, 5), 12) if head else (GATrVectorField, (8,), 10)
    ref, dense = _gatr_pair(cls, args, dense_linear=True)
    ref, dense = ref.double(), dense.double()
    x = torch.randn(3, n_in, 8, dtype=torch.float64)
    results = []
    for m in (ref, dense):
        xx = x.clone().requires_grad_()
        results.append(_grads(m, m(xx), [xx]))
    for a, b in zip(*results):
        torch.testing.assert_close(b, a, atol=1e-10, rtol=1e-10)


def _runs_compiled(module, x):
    """Whether a call of `module` goes through torch.compile (hooks run inside the compiled call)."""
    seen = []
    handle = module.register_forward_hook(lambda m, i, o: seen.append(torch.compiler.is_compiling()))
    module(x)
    handle.remove()
    return seen == [True]


def test_compiled_vector_field_matches_eager_and_its_copy_stays_compiled():
    import copy
    from pose3d.models.gatr_denoiser import GATrVectorField

    ref, comp = _gatr_pair(GATrVectorField, (8,), dense_linear=True, compile=True)
    x = torch.randn(3, 10, 8)
    torch.testing.assert_close(comp(x), ref(x), atol=1e-5, rtol=1e-4)
    assert _runs_compiled(comp, x) and not _runs_compiled(ref, x)
    # The EMA weights are a deepcopy (torch.optim.swa_utils.AveragedModel), which drops
    # Module.compile()'s compiled call: the copy must be compiled again and run its own weights.
    twin = copy.deepcopy(comp)
    assert _runs_compiled(twin, x)
    with torch.no_grad():
        for p in twin.parameters():
            p.mul_(0.5)
    ref_twin = copy.deepcopy(ref)
    ref_twin.load_state_dict(twin.state_dict())
    torch.testing.assert_close(twin(x), ref_twin(x), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(twin(x), comp(x))


def test_gatr_speed_flags_reach_the_vector_field():
    from gatr.layers.linear import EquiLinear
    from pose3d.config import parse_args
    from pose3d.models import build_model

    # On by default: they compute the same function as GATr's own layers.
    field = build_model(parse_args(["--path_to_datasets=/tmp", "--no-pretrained_backbone"]),
                        ALGEBRA).vector_field
    linears = [m for m in field.modules() if isinstance(m, EquiLinear)]
    assert linears and all(type(m) is not EquiLinear for m in linears)
    assert field.compiled
    off = parse_args(["--path_to_datasets=/tmp", "--no-gatr_dense_linear", "--no-gatr_compile",
                      "--no-pretrained_backbone"])
    plain = build_model(off, ALGEBRA).vector_field
    assert all(type(m) is EquiLinear for m in plain.modules() if isinstance(m, EquiLinear))
    assert not plain.compiled


def test_compile_leaves_room_for_every_block_copy_and_shape():
    # Each GATr block, the EMA copy, train / eval mode and every batch shape get their own
    # compiled frames; past torch's default limit of 8 dynamo silently falls back to eager.
    from pose3d.models.gatr_denoiser import GATrVectorField

    GATrVectorField(8, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2, compile=True)
    assert torch._dynamo.config.recompile_limit >= 64


def test_bf16_backbone_returns_fp32_close_to_the_fp32_backbone():
    from pose3d.models.clifford_flow import ImageToMultivectors

    torch.manual_seed(0)
    ref = ImageToMultivectors(ALGEBRA, pretrained_backbone=False, encoder_type="resnet50",
                              conv_adapter=False).eval()
    low = ImageToMultivectors(ALGEBRA, pretrained_backbone=False, encoder_type="resnet50",
                              conv_adapter=False, bf16=True).eval()
    low.load_state_dict(ref.state_dict())
    img = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        a, b = ref(img), low(img)
    assert b.dtype == torch.float32 and b.shape == a.shape
    assert (b - a).norm() / a.norm() < 0.05      # bf16 keeps ~3 significant digits
    assert (b - a).abs().max() > 0               # and it really ran in bf16


def test_backbone_bf16_flag_reaches_the_adapter():
    from pose3d.config import parse_args
    from pose3d.models import build_model

    on = parse_args(["--path_to_datasets=/tmp", "--no-pretrained_backbone"])  # the default
    off = parse_args(["--path_to_datasets=/tmp", "--no-pretrained_backbone", "--no-backbone_bf16"])
    assert build_model(on, ALGEBRA).adapter.bf16
    assert not build_model(off, ALGEBRA).adapter.bf16


def test_bf16_backbone_only_where_the_hardware_has_bf16(monkeypatch):
    from pose3d.models.clifford_flow import _native_bf16

    cuda = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # T4
    assert not _native_bf16(cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))  # A100
    assert _native_bf16(cuda)
    assert _native_bf16(torch.device("cpu"))
    assert not _native_bf16(torch.device("mps"))


def test_895_recipe_with_pose_harmonics_trains_and_samples():
    """The 8.95 recipe (pooled condition, Clifford condition head, GATr field over [rotor, t, condition]) with
    --gatr_pose_harmonics 6: every condition token gets 49 harmonics of its vector part in the pose's frame."""
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr")
    pytest.importorskip("e3nn")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.config import parse_args
    from pose3d.geometry.rotor import random_rotor, rotor_to_matrix
    from pose3d.models import build_model

    torch.manual_seed(0)
    base = ["--path_to_datasets", "x", "--vector_field", "gatr", "--no-pretrained_backbone", "--encoder", "resnet50",
            "--gatr_mv_channels", "4", "--gatr_s_channels", "8", "--gatr_blocks", "1", "--n_cond_mv", "4"]
    model = build_model(parse_args(base + ["--gatr_pose_harmonics", "6"]), CliffordAlgebra((1, 1, 1)))
    plain = build_model(parse_args(base), CliffordAlgebra((1, 1, 1)))
    assert model.vector_field.extra_scalars == 49 and plain.vector_field.extra_scalars == 0
    img = torch.randn(2, 3, 224, 224)
    loss = model.compute_loss(img, rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.eval().predict(img, n_samples=2, steps=2).shape == (2, 3, 3)
