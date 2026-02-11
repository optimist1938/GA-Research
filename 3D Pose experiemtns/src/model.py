# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
import torchvision
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from image2sphere.models import ResNet
from image_encoders import ImageEncoder
from image2sphere.so3_utils import so3_healpix_grid, flat_wigner, nearest_rotmat
from e3nn import o3
from typing import List,Union

def _so3_num_fourier_coeffs(lmax: int) -> int:
    return sum([(2 * l + 1) ** 2 for l in range(lmax + 1)])

class I2S(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        hidden_dim: List = [32],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self.encoder = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        enc_channels = getattr(self.encoder, "output_shape", None)[0]

        self._mv_dim = int(2**algebra.dim)
        self._n_mv = int(n_mv)

        self.project = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head = TralaleroTralala(
            algebra=algebra,
            in_features=self._n_mv,
            hidden_dim=hidden_dim,
            out_features=self.num_coeffs,
        )

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache",o3.angles_to_matrix(*self.so3_xyx),persistent=False)
        

    def forward(self, x: torch.tensor) -> torch.Tensor:
        fmap = self.encoder(x)
        fmap = self.avgpool(fmap).flatten(1)
        mv = self.project(fmap).view(fmap.shape[0], self._n_mv, self._mv_dim)
        coeffs_mv = self.ga_head(mv)
        coeffs = coeffs_mv[..., 0]
        logits = self.logits_on_grid(coeffs)
        logits = logits / max(self.temperature, 1e-8)
        return logits

    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)  

    @torch.no_grad()
    def probs_on_grid(self, logits : torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx
    
    @torch.no_grad()
    def predict(self, x) : 
        idx = self.predict_rotmat(self.forward(x))
        return self.so3_rotmats_cache[idx]
    
    @torch.no_grad()
    def get_nearest_idx(self, rot_gt : torch.Tensor) :
        return nearest_rotmat(rot_gt,self.so3_rotmats_cache)

class TralaleroTralala(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "gp": SteerableGeometricProductLayer(algebra, hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def forward(self, x):
        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["act2"](x)
        x = self.out(x)
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
