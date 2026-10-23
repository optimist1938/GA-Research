"""The spectral-bias probe on the sphere (probes/spectral_bias_sphere): the batched ensembles equal the library models, the pose-dependent
Clifford models are invariant, and the spectrum measurement returns 1 for the exact target."""
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("e3nn")
pytest.importorskip("clifford")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes" / "spectral_bias_sphere"))


@pytest.fixture(scope="module")
def mods():
    import spectral_bias_sphere as S
    import spectral_ens as EN
    return S, EN


@pytest.mark.parametrize("name", ["mlp", "cgenn"])
def test_ensemble_forward_and_adam_match_the_library_models(mods, name):
    S, EN = mods
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        refs = []
        for sd in (0, 1):
            torch.manual_seed(sd)
            refs.append(S.build(name).double())
        ens = (EN.EnsCGENN if name == "cgenn" else EN.EnsMLP)(refs).double()
        x = S.fibonacci_sphere(40, seed=3).double()
        xe = x.unsqueeze(0).expand(2, -1, -1).contiguous()
        with torch.no_grad():
            assert float((ens(xe) - torch.stack([r(x) for r in refs])).abs().max()) < 1e-10
        y = torch.randn(2, 40, dtype=torch.float64)
        opt = EN.EnsAdam(list(ens.parameters()), torch.tensor([1e-3, 1e-3], dtype=torch.float64))
        for _ in range(10):
            for p in ens.parameters():
                p.grad = None
            loss = (ens(xe) - y).pow(2).mean(-1)
            loss.sum().backward()
            opt.step()
        ref = refs[0]
        o = torch.optim.Adam(ref.parameters(), lr=1e-3)
        for _ in range(10):
            o.zero_grad()
            l = (ref(x) - y[0]).pow(2).mean()
            l.backward()
            o.step()
        assert abs(float(loss[0]) - float(l)) < 1e-10
    finally:
        torch.set_default_dtype(prev)


def test_exact_target_has_spectrum_one(mods):
    S, _ = mods
    L, LMAX = 4, 6
    blocks, a = S.make_target(0, L, [1.0] * L)
    xe = S.fibonacci_sphere(800, seed=2)
    pinv = torch.linalg.pinv(S.harmonics(xe, 0, LMAX))
    rho, amp = S.spectrum(lambda x: S.target_fn(x, blocks, a, L), xe, pinv, L, LMAX, blocks, a)
    assert all(abs(r - 1) < 1e-3 for r in rho) and all(abs(v - 1) < 1e-3 for v in amp)


def test_pose_models_are_invariant_under_a_joint_rotation(mods):
    S, _ = mods
    import spectral_bias_pose as P
    from e3nn import o3
    torch.manual_seed(0)
    x, R, _ = P.sample(6, 1)
    G = o3.rand_matrix(1).float()[0]
    names = ["cgenn"]
    try:
        from pose3d.models.gatr_denoiser import _ensure_xformers_stub, _import_gatr
        _ensure_xformers_stub()
        _import_gatr()
        names.append("gatr")
    except Exception:
        pass
    for name in names:
        m = P.build(name).eval()
        with torch.no_grad():
            assert float((m(x, R) - m(x @ G.T, G @ R)).abs().max()) < 1e-5, name
