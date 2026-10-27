"""Flow-trajectory visualization: geometry of the recorded paths and the no-op / no-side-effect guarantees of the W&B hook."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.engine import flow_viz
from pose3d.engine.trainer import log_viz
from pose3d.models.clifford_flow import CliffordFlow

ALGEBRA = CliffordAlgebra((1, 1, 1))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CliffordFlow(ALGEBRA, hidden_dim=[8], n_cond_mv=4, pretrained_backbone=False, encoder_type="resnet50",
                        conv_adapter=False).eval()


def _rot(n):
    q = torch.linalg.qr(torch.randn(n, 3, 3))[0]
    return (q * torch.linalg.det(q).sign().view(-1, 1, 1)).numpy()


def test_trajectories_start_at_noise_and_are_rotations(model):
    path = flow_viz.trajectories(model, torch.rand(3, 64, 64), k=5, steps=4, seed=1)
    assert path.shape == (5, 5, 3, 3)
    flat = path.reshape(-1, 3, 3)
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", flat, flat), np.tile(np.eye(3), (len(flat), 1, 1)), atol=1e-4)
    np.testing.assert_allclose(np.linalg.det(flat), 1.0, atol=1e-4)


def test_same_seed_same_noise_and_global_random_state_untouched(model):
    img = torch.rand(3, 64, 64)
    state = torch.random.get_rng_state().clone()
    a = flow_viz.trajectories(model, img, k=3, steps=2, seed=7)
    assert bool((state == torch.random.get_rng_state()).all())
    b = flow_viz.trajectories(model, img, k=3, steps=2, seed=7)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a[0], flow_viz.trajectories(model, img, k=3, steps=2, seed=8)[0])


def test_relative_vectors_are_zero_at_the_truth_and_bounded_by_pi():
    gt = _rot(1)[0]
    path = np.stack([np.stack([gt, _rot(1)[0]])] * 2)          # (2 times, 2 samples, 3, 3); sample 0 sits on the truth
    vec = flow_viz.relative_vectors(path, gt)
    np.testing.assert_allclose(vec[:, 0], 0, atol=1e-6)
    assert np.linalg.norm(vec, axis=-1).max() <= np.pi + 1e-6
    np.testing.assert_allclose(np.degrees(np.linalg.norm(vec[:, 1], axis=-1)), flow_viz.angle_deg(path[:, 1], gt), atol=1e-4)


def test_medoid_is_the_sample_closest_to_the_others():
    base = _rot(1)[0]
    near = [np.array(base) for _ in range(3)]
    far = _rot(1)[0]
    assert np.allclose(flow_viz._medoid(np.stack(near + [far])), base)


def test_figures_are_returned_without_a_path(model):
    imgs, rots = torch.rand(2, 3, 64, 64), _rot(2)
    fig = flow_viz.figure_paths("t", model, imgs, rots, [0, 1], [5.0, 9.0], k=3, steps=2, seed=0)
    assert fig is not None
    flow_viz.plt.close(fig)


def test_log_viz_is_a_noop_without_a_run_or_the_flag(model):
    ds = SimpleNamespace(__len__=lambda: 0)
    loader = SimpleNamespace(dataset=ds)
    cfg = SimpleNamespace(run=SimpleNamespace(log_viz=True, viz_images=2))
    assert log_viz(model, loader, None, cfg) is None                                           # no run
    cfg.run.log_viz = False
    assert log_viz(model, loader, SimpleNamespace(log=lambda *a, **k: pytest.fail("logged")), cfg) is None
    assert log_viz(object(), loader, SimpleNamespace(log=lambda *a, **k: pytest.fail("logged")),
                   SimpleNamespace(run=SimpleNamespace(log_viz=True, viz_images=2))) is None   # not a flow model


def test_hopf_coords_identify_rotation():
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from pose3d.engine.flow_viz import hopf_coords

    R = Rot.random(6, random_state=1).as_matrix()
    p, phi = hopf_coords(R)
    assert np.allclose(np.linalg.norm(p, axis=-1), 1)
    p2, phi2 = hopf_coords(R @ Rot.from_euler("z", 40, degrees=True).as_matrix())      # spin about the body z axis moves only the colour
    assert np.allclose(p, p2) and np.allclose((phi2 - phi) % 360, 40)
