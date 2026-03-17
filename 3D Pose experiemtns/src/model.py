# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from image_encoders import build_encoder
from image2sphere.so3_utils import so3_healpix_grid, flat_wigner, nearest_rotmat
from e3nn import o3
from typing import List, Union

def _so3_num_fourier_coeffs(lmax: int) -> int:
    return sum([(2 * l + 1) ** 2 for l in range(lmax + 1)])


def _move_tensors_in_object(obj, device: torch.device, visited: set):
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_tensors_in_object(value, device, visited)
        return obj

    if isinstance(obj, tuple):
        return tuple(_move_tensors_in_object(value, device, visited) for value in obj)

    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in list(vars(obj).items()):
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(obj, attr_name, moved_value)
        return obj

    return obj


def _move_unregistered_tensors_to_device(module: nn.Module, device: torch.device):
    visited = set()
    for submodule in module.modules():
        registered_params = set(dict(submodule.named_parameters(recurse=False)).keys())
        registered_buffers = set(dict(submodule.named_buffers(recurse=False)).keys())

        for attr_name, attr_value in list(submodule.__dict__.items()):
            if attr_name in registered_params or attr_name in registered_buffers:
                continue
            moved_value = _move_tensors_in_object(attr_value, device, visited)
            if moved_value is not attr_value:
                setattr(submodule, attr_name, moved_value)



class I2S(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        encoder_type: str = "resnet",
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self.encoder = build_encoder(encoder_type)
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

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        logits = self.forward(img)
        idx = self.get_nearest_idx(rot_gt).long().view(-1)
        return criterion(logits, idx)


class GA_I2S(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        n_mv: int = 8,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        encoder_type: str = "resnet",
        ga_pool_hw: tuple = (28, 28),
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        if len(ga_pool_hw) != 2:
            raise ValueError("ga_pool_hw must have exactly two elements: (height, width)")
        self.ga_pool_hw = (int(ga_pool_hw[0]), int(ga_pool_hw[1]))
        if self.ga_pool_hw[0] <= 0 or self.ga_pool_hw[1] <= 0:
            raise ValueError("ga_pool_hw values must be positive")

        self.pre_encode_pool = nn.AdaptiveAvgPool2d(self.ga_pool_hw)

        # Keep API compatibility with I2S but force canonical GA encoder.
        _ = encoder_type
        self.encoder = build_encoder("ga_canonical")

        enc_shape = getattr(self.encoder, "output_shape", None)
        if enc_shape is None or len(enc_shape) != 3:
            raise ValueError("GA encoder must expose output_shape = (mv_dim, h, w)")

        self._mv_dim = int(enc_shape[0])
        _ = n_mv
        self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])

        if self._mv_dim != int(2**algebra.dim):
            raise ValueError(
                f"Encoder multivector dim ({self._mv_dim}) must match algebra dim ({2**algebra.dim})"
            )
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
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        pooled_x = self.pre_encode_pool(x)
        mv_grid = self.encoder(pooled_x)
        b, mv_dim, h, w = mv_grid.shape

        mv = mv_grid.permute(0, 2, 3, 1).reshape(b, h * w, mv_dim)

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
    def probs_on_grid(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx

    @torch.no_grad()
    def predict(self, x):
        idx = self.predict_rotmat(self.forward(x))
        return self.so3_rotmats_cache[idx]

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        logits = self.forward(img)
        idx = self.get_nearest_idx(rot_gt).long().view(-1)
        return criterion(logits, idx)

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
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra, encoder_type: str = "resnet", ga_pool_hw: tuple = (28, 28)):
        super().__init__()
        self.algebra = algebra
        self._use_ga_backbone = encoder_type in {"ga", "ga_canonical"}
        self._mv_dim = int(2**algebra.dim)

        if self._use_ga_backbone:
            if len(ga_pool_hw) != 2:
                raise ValueError("ga_pool_hw must have exactly two elements: (height, width)")
            self.ga_pool_hw = (int(ga_pool_hw[0]), int(ga_pool_hw[1]))
            if self.ga_pool_hw[0] <= 0 or self.ga_pool_hw[1] <= 0:
                raise ValueError("ga_pool_hw values must be positive")

            self.pre_encode_pool = nn.AdaptiveAvgPool2d(self.ga_pool_hw)
            self.backbone = build_encoder(encoder_type)
            self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])
        else:
            self._n_mv = 8
            self.backbone = build_encoder(encoder_type)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            enc_channels = getattr(self.backbone, "output_shape", None)[0]
            self.projective_matrix = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.ga_head = TralaleroTralala(algebra, in_features=self._n_mv)

    def _ga_to_canonical_mv(self, mv_grid: torch.Tensor) -> torch.Tensor:
        if mv_grid.shape[1] == self._mv_dim:
            return mv_grid

        if mv_grid.shape[1] != 6:
            raise ValueError(
                f"Unsupported GA encoder channels: expected 6 or {self._mv_dim}, got {mv_grid.shape[1]}"
            )

        e, e123, e1, e2, e13, e23 = torch.unbind(mv_grid, dim=1)
        zeros = torch.zeros_like(e)
        return torch.stack([e, e1, e2, zeros, zeros, e13, e23, e123], dim=1)


    def forward(self, x):
        if self._use_ga_backbone:
            pooled_x = self.pre_encode_pool(x)
            mv_grid = self.backbone(pooled_x)
            mv_grid = self._ga_to_canonical_mv(mv_grid)
            b, mv_dim, h, w = mv_grid.shape
            x = mv_grid.permute(0, 2, 3, 1).reshape(b, h * w, mv_dim)
        else:
            x = self.backbone(x)
            x = self.avgpool(x)
            x = x.flatten(1, -1)
            x = self.projective_matrix(x)
            x = x.reshape(x.shape[0], self._n_mv, self._mv_dim)

        x = self.ga_head(x)
        x = x[:, :, 0]
        x = x.reshape(x.shape[0], 3, 3)
        return x



class MLPBaseline(nn.Module):
    def __init__(self, encoder_type: str = "resnet"):
        super().__init__()
        self.backbone = build_encoder(encoder_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        enc_channels = getattr(self.backbone, "output_shape", None)[0]
        self.linear_head = nn.Linear(in_features=enc_channels, out_features=9)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.linear_head(x)
        x = x.reshape(x.shape[0], 3, 3)
        return x
