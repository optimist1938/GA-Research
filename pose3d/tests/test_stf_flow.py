"""The two-tower CliffordFlow (stf_track): equivariance against the control, wiring, compile, sizes.

Needs the GATr package (skipped without it, as test_gatr_denoiser.py). The tensor algebra itself is in
test_stf_track.py.
"""

import pytest
import torch

from pose3d.models.stf_track import stf_feature_blocks

from pose3d.models.gatr_denoiser import _ensure_xformers_stub

_ensure_xformers_stub()
pytest.importorskip("gatr", reason="GATr package not installed")
from clifford.algebra.cliffordalgebra import CliffordAlgebra  # noqa: E402

from pose3d.geometry.rotor import embed_rotor, random_rotor, rotor_to_matrix  # noqa: E402
from pose3d.models.clifford_flow import CliffordFlow, StfCondition  # noqa: E402

ALGEBRA = CliffordAlgebra((1, 1, 1))
GATR = dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)


def flow(stf=False, lmax=3, inject="gram", gatr=GATR, **kw):
    torch.manual_seed(0)
    return CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                        vector_field="gatr", gatr=dict(gatr), stf_track=stf, stf_lmax=lmax,
                        stf_channels=4, stf_inject=inject, **kw)


def randomize_heads(model):
    """Non-zero weights everywhere past the backbone (the read-out starts at zero)."""
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith("adapter.backbone."):
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))


def sandwich(g, x):
    """g x g^-1 for rotors g (B, 4) on multivectors x (B, ..., 8) of Cl(3,0)."""
    gm = embed_rotor(g, ALGEBRA)
    gi = ALGEBRA.inverse(gm)
    shape = x.shape
    flat = x.reshape(shape[0], -1, 8)
    gm_e, gi_e = gm[:, None].expand_as(flat), gi[:, None].expand_as(flat)
    return ALGEBRA.sandwich(gm_e, flat, gi_e).reshape(shape)


def conj_rotor(r, g):
    """The rotor of G R G^T: the sandwich of r by g, back as 4 numbers."""
    return sandwich(g, embed_rotor(r, ALGEBRA).unsqueeze(1))[:, 0][:, [0, 4, 5, 6]]


def equivariance_error(model, seed=0):
    """Conjugate the condition multivectors and the rotor by one rotor g: the velocity must be conjugated too
    (GATr's symmetry, g x g^-1 on every token). Returns max |v(g.) - g v(.)|."""
    torch.manual_seed(seed)
    b = 3
    raw = torch.randn(b, 8, 8)
    r, g = random_rotor(b), random_rotor(b)
    t = torch.rand(b)
    model.eval()
    with torch.no_grad():
        v = model.velocity(r, t, model._couple(raw))
        v_g = model.velocity(conj_rotor(r, g), t, model._couple(sandwich(g, raw)))
        v8 = torch.zeros(b, 8)
        v8[:, 4:7] = v                          # the velocity is the bivector coefficients e12, e13, e23
        expect = sandwich(g, v8.unsqueeze(1))[:, 0, 4:7]
    return (v_g - expect).abs().max().item(), v.abs().max().item()


def test_control_is_unchanged_when_stf_is_off():
    model = flow(stf=False)
    assert model.stf_tower is None and model.vector_field.stf_embed is None
    assert not any("stf" in k for k in model.state_dict())
    assert model.vector_field.net.linear_in.s2s.in_features == 2     # t and the query flag only


def test_control_equivariance_reference():
    """The control model is equivariant under conjugation of tokens (GATr's sandwich symmetry)."""
    model = flow(stf=False)
    randomize_heads(model)
    err, scale = equivariance_error(model)
    assert scale > 1e-3 and err < 1e-4 * max(1.0, scale)


@pytest.mark.parametrize("lmax", [2, 3, 6])
def test_two_tower_model_has_the_control_equivariance(lmax):
    model = flow(stf=True, lmax=lmax, inject="gram")
    randomize_heads(model)
    err, scale = equivariance_error(model)
    assert scale > 1e-3, "the velocity must be non-trivial for the test to mean anything"
    assert err < 1e-4 * max(1.0, scale), (err, scale)


def test_the_stf_scalars_change_the_velocity_and_depend_on_the_rotor_at_high_degree():
    model = flow(stf=True, lmax=3)
    randomize_heads(model)
    torch.manual_seed(1)
    cond = model._couple(torch.randn(2, 8, 8))
    r, t = random_rotor(2), torch.rand(2)
    with torch.no_grad():
        v = model.velocity(r, t, cond)
        zeroed = StfCondition(cond.mv, tuple(torch.zeros_like(s) for s in cond.stf))
        assert (model.velocity(r, t, zeroed) - v).abs().max() > 5e-3 * v.abs().max()   # the STF track reaches the velocity


def test_frame_mode_is_left_invariant_at_the_injection_but_not_conjugation_equivariant():
    model = flow(stf=True, lmax=3, inject="frame")
    randomize_heads(model)
    err, scale = equivariance_error(model)
    assert err > 1e-4 * max(1.0, scale)    # documented: frame mode gives up the control's symmetry


def test_two_tower_trains_samples_and_backpropagates_to_both_towers():
    model = flow(stf=True, lmax=3)
    randomize_heads(model)
    model.train()
    img = torch.randn(2, 3, 64, 64)
    rot = rotor_to_matrix(random_rotor(2), ALGEBRA)
    loss = model.compute_loss(img, rot)
    loss.backward()
    assert torch.isfinite(loss)
    for part in (model.stf_tower, model.vector_field.stf_embed, model.condition_head):
        grads = [p.grad for p in part.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads)
    out = model.eval().predict(img, n_samples=2, steps=3)
    assert out.shape == (2, 3, 3) and torch.isfinite(out).all()


def test_zero_initialised_readout_gives_zero_velocity_as_before():
    model = flow(stf=True, lmax=3)
    cond = model._couple(torch.randn(2, 8, 8))
    assert model.velocity(random_rotor(2), torch.rand(2), cond).abs().max() == 0


def test_stf_track_rejects_unsupported_combinations():
    with pytest.raises(ValueError):
        CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="clifford", stf_track=True)
    with pytest.raises(ValueError):
        CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="gatr", stf_track=True, mlp_heads=True)
    with pytest.raises(ValueError):
        CliffordFlow(ALGEBRA, hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False, encoder_type="resnet50",
                     conv_adapter=False, vector_field="gatr", stf_track=True, pose_tokens="frame",
                     gatr=dict(GATR))


def test_stf_flags_reach_the_model_and_validate():
    from pose3d.config import parse_args
    from pose3d.models import build_model

    cfg = parse_args(["--path_to_datasets=/tmp", "--no-pretrained_backbone", "--encoder", "resnet50",
                      "--stf_track", "--stf_lmax", "6", "--stf_channels", "4", "--stf_inject", "frame",
                      "--no-gatr_compile"])
    model = build_model(cfg, ALGEBRA)
    assert model.stf_tower.lmax == 6 and model.stf_tower.channels == 4 and model.stf_inject == "frame"
    bad = parse_args(["--path_to_datasets=/tmp", "--stf_track", "--stf_lmax", "7"])
    with pytest.raises(ValueError):
        build_model(bad, ALGEBRA)
    default = parse_args(["--path_to_datasets=/tmp"])
    assert default.flow.stf_track is False


def test_compiled_two_tower_vector_field_matches_eager():
    from pose3d.models.gatr_denoiser import GATrVectorField

    torch.manual_seed(0)
    blocks = stf_feature_blocks(3, 4, "gram")
    ref = GATrVectorField(8, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2, stf_blocks=blocks)
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.3)
    comp = GATrVectorField(8, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2, stf_blocks=blocks,
                           dense_linear=True, compile=True)
    comp.load_state_dict(ref.state_dict())
    x, f = torch.randn(3, 10, 8), torch.randn(3, sum(blocks))
    torch.testing.assert_close(comp(x, stf_features=f), ref(x, stf_features=f), atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        ref(x)


def non_backbone_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("adapter.backbone."))


def test_parameter_counts_of_the_two_arms_at_the_reference_sizes(capsys):
    """Prints the non-backbone parameter counts of the control and of the two-tower model (reference sizes:
    flow_hidden_dim 88, n_cond_mv 64, GATr 4 blocks x 8 mv x 32 s channels), for the run log."""
    from pose3d.config import parse_args
    from pose3d.models import build_model

    counts = {}
    for name, extra in [("control", []), ("stf_lmax3", ["--stf_track"]),
                        ("stf_lmax6", ["--stf_track", "--stf_lmax", "6"])]:
        cfg = parse_args(["--path_to_datasets=/tmp", "--no-pretrained_backbone", "--encoder", "resnet50",
                          "--no-gatr_compile", *extra])
        counts[name] = non_backbone_params(build_model(cfg, ALGEBRA))
    with capsys.disabled():
        print("\nnon-backbone parameters:", counts)
    assert counts["control"] < counts["stf_lmax3"] < counts["stf_lmax6"]


def test_two_tower_model_runs_the_symsol_evaluation_and_likelihood(tmp_path):
    """The SYMSOL scoring path (nearest-equivalent error, then the ODE log-likelihood with a divergence by
    autograd through the velocity) accepts the two-tower condition, a tuple of tensors."""
    import numpy as np
    from torch.utils.data import DataLoader

    from test_symsol import _fake_symsol
    from pose3d.datasets.symsol import Symsol
    from pose3d.engine.likelihood import symsol_log_likelihood
    from pose3d.engine.metrics import calculate_evaluation_metrics

    loader = DataLoader(Symsol(_fake_symsol(str(tmp_path), n_views=1), train=False), batch_size=3)
    model = flow(stf=True, lmax=3)
    randomize_heads(model)
    model.eval()

    class Cfg:
        device = torch.device("cpu")
        rank = 0
        features = type("F", (), {"fixed_val_noise": True})()
        run = type("R", (), {"platform": "kaggle"})()

    err, cls = calculate_evaluation_metrics(model, loader, Cfg(), n_samples=2, return_classes=True)
    assert err.shape == (5,) and np.isfinite(err).all() and cls.tolist() == [0, 1, 2, 3, 4]
    ll, _ = symsol_log_likelihood(model, loader, "cpu", n_equivalents=2, steps=2)
    assert ll.shape == (5,) and np.isfinite(ll).all()


def test_symsol_likelihood_runs_through_the_compiled_vector_field(tmp_path):
    """The log-likelihood differentiates the velocity three times with retain_graph (the divergence); a
    torch.compiled vector field refuses that ("compiled with non-empty donated buffers"), which killed the
    first SYMSOL RTX runs at the final evaluation. The likelihood must run the compiled model eagerly."""
    import numpy as np
    from torch.utils.data import DataLoader

    from test_symsol import _fake_symsol
    from pose3d.datasets.symsol import Symsol
    from pose3d.engine.likelihood import symsol_log_likelihood

    loader = DataLoader(Symsol(_fake_symsol(str(tmp_path), n_views=1), train=False), batch_size=5)
    model = flow(stf=False, gatr=dict(GATR, dense_linear=True, compile=True))
    randomize_heads(model)
    model.eval()
    ll, _ = symsol_log_likelihood(model, loader, "cpu", n_equivalents=2, steps=2)
    assert ll.shape == (5,) and np.isfinite(ll).all()
