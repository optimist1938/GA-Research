import unittest

import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from src.model import SimpleFullGeometricProductPoseHead


class ViTGARotorReadoutSmokeTest(unittest.TestCase):
    def test_rotor_readout_returns_rotation_matrix(self):
        algebra = CliffordAlgebra((1, 1, 1))
        head = SimpleFullGeometricProductPoseHead(
            algebra=algebra,
            input_dim=512,
            input_features=256,
            hidden_dim=-1,
            readout_type="rotor",
        )

        output = head(torch.randn(2, 256, 512))

        self.assertEqual(output.shape, (2, 3, 3))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_rotor_readout_requires_cl3(self):
        with self.assertRaisesRegex(
            ValueError,
            "vit_ga_readout_type='rotor' requires algebra_dim=3 / Cl\\(3,0\\)",
        ):
            SimpleFullGeometricProductPoseHead(
                algebra=CliffordAlgebra((1, 1, 1, 1)),
                input_dim=512,
                input_features=256,
                hidden_dim=-1,
                readout_type="rotor",
            )


if __name__ == "__main__":
    unittest.main()
