import pytest
import torch

pytest.importorskip("image2sphere")


def test_target_is_the_grid_function_at_the_truth():
    """<D(R_gt), D(R)> peaks at R = R_gt, so the regression target decodes to the truth through the same grid inference uses."""
    from e3nn import o3
    from image2sphere import so3_utils
    from pose3d.models.wignerd import WignerD

    xyx = so3_utils.so3_healpix_grid(rec_level=2)
    grid_w = so3_utils.flat_wigner(6, *xyx)                     # (Q, M)
    grid_R = o3.angles_to_matrix(*xyx)
    stub = WignerD.__new__(WignerD)
    stub.lmax = 6
    for i in (0, 37, 211):
        R = grid_R[i:i + 1]
        t = WignerD.target(stub, R)
        assert t.shape == (1, 455)
        assert int((grid_w @ t[0]).argmax()) == i


def test_loss_weights_cover_every_coefficient():
    from pose3d.models.wignerd import degree_weights

    unit, uniform = degree_weights(6, "unit"), degree_weights(6, "uniform")
    assert unit.numel() == uniform.numel() == 455
    assert unit[0] == 1.0 and abs(unit[-1].item() - 1 / 169) < 1e-9 and bool((uniform == 1).all())


def test_forward_loss_backward_and_predict():
    from pose3d.models.wignerd import WignerD
    from scipy.spatial.transform import Rotation

    m = WignerD(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=1)
    img = torch.rand(2, 3, 224, 224)
    rot = torch.tensor(Rotation.random(2, random_state=0).as_matrix(), dtype=torch.float32)
    loss = m.compute_loss(img, rot)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert m.predict(img).shape == (2, 3, 3)


def test_target_accepts_slightly_non_orthonormal_rotations():
    """Warp-augmented targets are only approximately rotations; e3nn's matrix_to_angles asserts det == 1."""
    from scipy.spatial.transform import Rotation
    from pose3d.models.wignerd import WignerD

    stub = WignerD.__new__(WignerD)
    stub.lmax = 6
    R = torch.tensor(Rotation.random(5, random_state=3).as_matrix(), dtype=torch.float32)
    noisy = R + 2e-3 * torch.randn_like(R)
    t_clean, t_noisy = WignerD.target(stub, R), WignerD.target(stub, noisy)
    assert torch.isfinite(t_noisy).all() and (t_clean - t_noisy).abs().max() < 0.1


def test_mean_reduction_divides_the_sum_by_the_coefficient_count():
    from pose3d.models.wignerd import WignerD

    a = WignerD(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=1, loss_reduction="sum")
    b = WignerD(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=1, loss_reduction="mean", loss_weights="unit")
    assert torch.allclose(a.loss_w / 455, b.loss_w)
    c = WignerD(encoder_type="resnet50", pretrained_backbone=False, eval_rec_level=1, loss_reduction="mean", loss_weights="uniform")
    assert torch.allclose(c.loss_w, torch.full((455,), 1 / 455))      # F.mse_loss over the 455 coefficients
