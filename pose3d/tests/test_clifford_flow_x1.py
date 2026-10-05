"""The endpoint (x1) parametrisation of CliffordFlow: --flow_param x1 [--x1_loss tangent|geodesic]."""

import unittest

import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.geometry.flow import geodesic_interpolate, relative_log
from pose3d.geometry.rotor import random_rotor, rotor_to_matrix
from pose3d.models.clifford_flow import CliffordFlow

ALGEBRA = CliffordAlgebra((1, 1, 1))


def _tiny_flow(**kwargs):
    torch.manual_seed(0)
    # A random (not pretrained) ResNet-50 and tiny GA heads: nothing is downloaded and the
    # whole model runs on CPU in seconds.
    return CliffordFlow(ALGEBRA, hidden_dim=[8], n_cond_mv=4, n_time_samples=2,
                        pretrained_backbone=False, encoder_type="resnet50",
                        conv_adapter=False, **kwargs)


def _randomize(module):
    """The vector field's output layer is zero-initialised, which would make every check
    below trivially true; give it random weights."""
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.1)


class RemainingDisplacementTest(unittest.TestCase):
    def test_scaled_velocity_is_the_log_from_the_interpolated_rotor(self):
        torch.manual_seed(1)
        r0, r1, t = random_rotor(64), random_rotor(64), torch.rand(64)
        rt = geodesic_interpolate(r0, r1, t, ALGEBRA)
        scaled = (1 - t).unsqueeze(-1) * relative_log(r0, r1, ALGEBRA)
        remaining = relative_log(rt, r1, ALGEBRA)
        self.assertTrue(torch.allclose(scaled, remaining, atol=1e-5))


class X1ParametrisationTest(unittest.TestCase):
    def test_velocity_divides_the_field_by_one_minus_t(self):
        model = _tiny_flow(flow_param="x1")
        _randomize(model.vector_field)
        cond = torch.randn(5, 4, 8)
        rotor, t = random_rotor(5), 0.9 * torch.rand(5)
        v = model.velocity(rotor, t, cond)
        b_hat = model._field(rotor, t, cond)
        self.assertEqual(b_hat.shape, (5, 3))
        self.assertTrue(torch.allclose(v, b_hat / (1 - t).unsqueeze(-1), atol=1e-6))
        # t = 1 is never reached by the sampler, but the clamp keeps it finite.
        self.assertTrue(torch.isfinite(model.velocity(rotor, torch.ones(5), cond)).all())

    def test_x1_modes_train_and_predict(self):
        variants = (
            dict(flow_param="x1"),
            dict(flow_param="x1", x1_loss="geodesic"),
            dict(flow_param="x1", mlp_heads=True),
        )
        for kwargs in variants:
            with self.subTest(**kwargs):
                model = _tiny_flow(**kwargs)
                _randomize(model.vector_field)
                img = torch.rand(2, 3, 64, 64)
                rot = rotor_to_matrix(random_rotor(2), ALGEBRA)

                loss = model.compute_loss(img, rot)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                grads = [p.grad for p in model.vector_field.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

                model.eval()
                pred = model.predict(img, n_samples=2, steps=3)
                self.assertEqual(pred.shape, (2, 3, 3))
                eye = torch.eye(3).expand(2, 3, 3)
                self.assertTrue(torch.allclose(pred @ pred.transpose(-1, -2), eye, atol=1e-4))
                self.assertTrue(torch.allclose(torch.det(pred), torch.ones(2), atol=1e-4))

    def test_tangent_loss_is_the_velocity_loss_reweighted(self):
        # With b_hat = 0 the tangent loss is E[(1-t)^2 |B|^2] and the velocity loss E[|B|^2];
        # the same (r0, r1, t) draws give exactly the reweighted value.
        model = _tiny_flow(flow_param="x1")
        cond = torch.zeros(6, 4, 8)
        torch.manual_seed(2)
        r0, r1, t = random_rotor(6), random_rotor(6), torch.rand(6)
        rt = geodesic_interpolate(r0, r1, t, ALGEBRA)
        b_hat = model._field(rt, t, cond)
        self.assertTrue(torch.equal(b_hat, torch.zeros(6, 3)))    # zero-initialised output
        target = (1 - t).unsqueeze(-1) * relative_log(r0, r1, ALGEBRA)
        expected = target.pow(2).sum(-1).mean()
        self.assertTrue(torch.allclose(
            ((1 - t) ** 2 * relative_log(r0, r1, ALGEBRA).pow(2).sum(-1)).mean(), expected))

    def test_default_is_the_velocity_parametrisation(self):
        model = _tiny_flow()
        self.assertEqual(model.flow_param, "velocity")
        _randomize(model.vector_field)
        cond = torch.randn(3, 4, 8)
        rotor, t = random_rotor(3), torch.rand(3)
        self.assertTrue(torch.equal(model.velocity(rotor, t, cond), model._field(rotor, t, cond)))

    def test_rejects_unknown_options(self):
        with self.assertRaises(ValueError):
            _tiny_flow(flow_param="endpoint")
        with self.assertRaises(ValueError):
            _tiny_flow(flow_param="x1", x1_loss="chordal")


if __name__ == "__main__":
    unittest.main()
