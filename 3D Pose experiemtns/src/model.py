# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
import torchvision
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from image2sphere.models import ResNet
from image2sphere.so3_uitls import so3_healpix_grid, flat_wigner, nearest_rotmat
from e3nn import o3
from typing import Dict

def _so3_num_fourier_coeffs(lmax: int) -> int:
    return sum([(2 * l + 1) ** 2 for l in range(lmax + 1)])


class CliffordFourierHead(nn.Module):
    """CG/GA head that maps a set of multivectors to scalar Fourier coefficients."""

    def __init__(self, algebra, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.fcgp1 = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=in_features, out_features=hidden_dim
        )
        self.act = MVSiLU(algebra, hidden_dim)
        self.gp = SteerableGeometricProductLayer(algebra, hidden_dim)
        self.fcgp2 = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=hidden_dim, out_features=out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features, mv_dim)
        x = self.fcgp1(x)
        x = self.act(x)
        x = self.gp(x)
        x = self.act(x)
        x = self.fcgp2(x)
        return x


class I2S(nn.Module):
    """Image2Sphere-like probabilistic pose head, but with Clifford/GA blocks.

    Forward returns Fourier coefficients for a density s: SO(3) -> R in the SO(3) Fourier basis
    (flattened Wigner-D blocks up to lmax). Provides helpers to evaluate the density on an SO(3)
    grid and a probabilistic loss (cross-entropy over SO(3) grid bins).
    """

    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        mv_dim: int = 32,
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self.encoder = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        enc_channels = getattr(self.encoder, "output_shape", None)

        self._mv_dim = int(mv_dim)
        self._n_mv = int(n_mv)

        self.project = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head = CliffordFourierHead(
            algebra=algebra,
            in_features=self._n_mv,
            hidden_dim=hidden_dim,
            out_features=self.num_coeffs,
        )

        # Precompute SO(3) grid + Wigner matrices on CPU; register as buffers
        xyx = so3_healpix_grid(rec_level=self.rec_level)  # (3, N)
        wign = flat_wigner(self.lmax, *xyx)  # (N, num_coeffs)
        # store for fast matmul: coeffs @ wigner_T -> logits over grid
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("_so3_rotmats_cache",o3.angles_to_matrix(*self.so3_xyx),persistent=False)  # (K, N)
        

    def forward(self, x: torch.tensor) -> torch.Tensor:
        fmap = self.encoder(x)              # (B, C, H, W)
        fmap = self.avgpool(fmap).flatten(1)  # (B, C)
        mv = self.project(fmap).view(fmap.shape[0], self._n_mv, self._mv_dim)  # (B, n_mv, mv_dim)

        coeffs_mv = self.ga_head(mv)        # (B, K, mv_dim) multivector outputs
        coeffs = coeffs_mv[..., 0]          # scalar part -> (B, K)
        return coeffs

    @torch.no_grad()
    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Evaluate (unnormalized) density logits on the internal SO(3) grid."""
        # coeffs: (B, K) or (B, 1, K)
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)  # (B, N)

    @torch.no_grad()
    def probs_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        logits = self.logits_on_grid(coeffs) / max(self.temperature, 1e-8)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Return MAP rotation (closest grid rotation)."""
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)  # (B,)
        # convert grid angles -> rotation matrices lazily via e3nn in so3_utils? not provided;
        # we return indices to avoid adding more deps here.
        return idx

    def loss(
        self,
        data: Dict[str, torch.Tensor],
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Probabilistic loss like I2S: CrossEntropy over SO(3) grid bins.

        - Predict Fourier coeffs
        - Evaluate logits on SO(3) grid via Wigner matrices
        - Find nearest grid bin to GT rotation
        - CrossEntropy(logits, idx)
        """
        img = data["img"]
        rot_gt = data["rot"]

        coeffs = self.forward(img)                    # (B, K)
        logits = self.logits_on_grid(coeffs)          # (B, N)
        logits = logits / max(self.temperature, 1e-8)

        if self._so3_rotmats_cache.device != logits.device:
            self._so3_rotmats_cache.device = self._so3_rotmats_cache.device.to(logits.device)

        idx = nearest_rotmat(rot_gt, self._so3_rotmats_cache)  # (B,)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, idx)

class TralaleroTralala(nn.Module):
    def __init__(self, algebra, in_features=512, hidden_dim=32, out_features=9):
        super().__init__()
        self.fcgp1 = FullyConnectedSteerableGeometricProductLayer(algebra, in_features=in_features, out_features=hidden_dim)
        self.activ = MVSiLU(algebra, hidden_dim)
        self.gp1 = SteerableGeometricProductLayer(algebra, hidden_dim)
        self.gp2 = FullyConnectedSteerableGeometricProductLayer(algebra, hidden_dim, out_features)

    def forward(self, x):
        x = self.fcgp1(x)
        x = self.activ(x)
        x = self.gp1(x)
        x = self.activ(x)
        x = self.gp2(x)
        return x


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.backbone = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projective_matrix = nn.Linear(2048, 32 * 8)
        self.ga_head = TralaleroTralala(algebra, in_features=8)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.projective_matrix(x)
        x = x.reshape(x.shape[0], -1, 32)
        x = self.ga_head(x)
        x = x[:, :, 0]
        x = x.reshape(x.shape[0], 3, 3)
        return x



class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_head = nn.Linear(in_features=2048, out_features=9)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.linear_head(x)
        x = x.reshape(x.shape[0], 3, 3)
        return x
