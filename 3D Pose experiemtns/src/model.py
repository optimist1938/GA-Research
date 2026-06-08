# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
import torchvision
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from clifford.models.modules.linear import MVLinear
from image_encoders import build_encoder
from image2sphere.so3_utils import so3_healpix_grid, flat_wigner, nearest_rotmat
from e3nn import o3
from typing import List,Union


def _unit_quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape [B, 4], got {tuple(q.shape)}")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1),
            torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1),
            torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ],
        dim=-2,
    )


def _project_multivector_to_rotor(mv: torch.Tensor) -> torch.Tensor:
    if mv.ndim != 2 or mv.shape[-1] != 8:
        raise ValueError(f"Expected multivector shape [B, 8], got {tuple(mv.shape)}")
    rotor = mv[:, [0, 4, 5, 6]]
    rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return rotor


def _mv_to_rotation_matrix(mv: torch.Tensor) -> torch.Tensor:
    if mv.ndim != 2 or mv.shape[-1] != 8:
        raise ValueError(f"Expected multivector shape [B, 8], got {tuple(mv.shape)}")

    v0 = mv[:, 0]
    v4 = mv[:, 4]
    v5 = mv[:, 5]
    v6 = mv[:, 6]

    norm = torch.sqrt((v0 * v0 + v4 * v4 + v5 * v5 + v6 * v6).clamp_min(1e-8))
    r0 = v0 / norm
    r4 = v4 / norm
    r5 = v5 / norm
    r6 = v6 / norm

    m = torch.zeros((mv.shape[0], 3, 3), dtype=mv.dtype, device=mv.device)
    m[:, 0, 0] = 1.0 - 2.0 * (r4 * r4 + r5 * r5)
    m[:, 0, 1] = 2.0 * (r0 * r4 - r5 * r6)
    m[:, 0, 2] = 2.0 * (r0 * r5 + r4 * r6)
    m[:, 1, 0] = -2.0 * (r0 * r4 + r5 * r6)
    m[:, 1, 1] = 1.0 - 2.0 * (r4 * r4 + r6 * r6)
    m[:, 1, 2] = 2.0 * (r0 * r6 - r4 * r5)
    m[:, 2, 0] = 2.0 * (r4 * r6 - r0 * r5)
    m[:, 2, 1] = 2.0 * (r0 * r6 + r4 * r5)
    m[:, 2, 2] = 1.0 - 2.0 * (r6 * r6 + r5 * r5)
    return m


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
                    "fc": MVLinear(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "gp": SteerableGeometricProductLayer(algebra, hd),
                    "ln": MVLayerNorm(algebra, hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = MVLinear(
            algebra, in_features=prev, out_features=out_features
        )

        print(self)

    def forward(self, x):
        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["ln"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x



class TransformerLikeMVBlock(nn.Module):
    def __init__(self, algebra, n_multivectors: int):
        super().__init__()
        if int(n_multivectors) <= 0:
            raise ValueError("n_multivectors must be positive")
        n_multivectors = int(n_multivectors)

        self.attn_like = FullyConnectedSteerableGeometricProductLayer(
            algebra,
            in_features=n_multivectors,
            out_features=n_multivectors,
        )
        self.up = MVLinear(algebra, n_multivectors, 4 * n_multivectors)
        self.act = MVSiLU(algebra, 4 * n_multivectors)
        self.down = MVLinear(algebra, 4 * n_multivectors, n_multivectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attn_like(x)
        x_ffn = self.up(attended)
        x_ffn = self.act(x_ffn)
        x_ffn = self.down(x_ffn)
        return attended + x_ffn


class TransformerLikeMVHead(nn.Module):
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

        self.blocks = nn.ModuleList(
            [TransformerLikeMVBlock(algebra, n_multivectors=in_features) for _ in hidden_dims]
        )
        self.out = MVLinear(algebra, in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class ResidualGeometricProductBlock(nn.Module):
    def __init__(
        self,
        algebra,
        n_multivectors: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        if int(n_multivectors) <= 0:
            raise ValueError("n_multivectors must be positive")

        self.algebra = algebra
        self.n_multivectors = int(n_multivectors)

        self.norm = nn.LayerNorm(2 ** algebra.dim) if use_layer_norm else nn.Identity()

        self.linear_in = MVLinear(algebra, self.n_multivectors, self.n_multivectors)
        self.act1 = MVSiLU(algebra, self.n_multivectors)
        self.gp = SteerableGeometricProductLayer(algebra, self.n_multivectors)
        self.act2 = MVSiLU(algebra, self.n_multivectors)
        self.linear_out = MVLinear(algebra, self.n_multivectors, self.n_multivectors)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.norm(x)
        y = self.linear_in(y)
        y = self.act1(y)
        y = self.gp(y)
        y = self.act2(y)
        y = self.linear_out(y)
        y = self.dropout(y)

        return residual + y


class ResidualGeometricProductHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
        num_blocks: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_blocks = int(num_blocks)

        if self.in_features <= 0:
            raise ValueError("in_features must be positive")

        if self.out_features <= 0:
            raise ValueError("out_features must be positive")

        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # Keep hidden_dim only for API compatibility with TralaleroTralala.
        self.hidden_dim = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualGeometricProductBlock(
                    algebra=algebra,
                    n_multivectors=self.in_features,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = MVLinear(algebra, self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for block in self.blocks:
            x = block(x)

        return self.out(x)


class ReducedGeometricProductHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
        mixing_layer: str = "gp",
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.mixing_layer = str(mixing_layer).lower()
        if self.mixing_layer not in {"gp", "mvlinear", "linear"}:
            raise ValueError("mixing_layer must be one of: gp, mvlinear, linear")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "mix": self._build_mixing_layer(algebra=algebra, channels=hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def _build_mixing_layer(self, algebra, channels: int):
        if self.mixing_layer == "gp":
            return SteerableGeometricProductLayer(algebra, channels)
        if self.mixing_layer == "mvlinear":
            return MVLinear(algebra, channels, channels)
        if self.mixing_layer == "linear":
            return nn.Linear(channels * (2**algebra.dim), channels * (2**algebra.dim))
        raise ValueError(f"Unsupported mixing_layer: {self.mixing_layer}")

    def _apply_mixing(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.mixing_layer == "linear":
            b, n, d = x.shape
            x = x.reshape(b, n * d)
            x = layer(x)
            return x.reshape(b, n, d)
        return layer(x)

    def forward(self, x):
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = self._apply_mixing(b["mix"], x)
            x = b["act2"](x)
        x = self.out(x)
        return x


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra, encoder_type: str = "resnet", ga_pool_hw: tuple = (28, 28), hidden_dim = [32]):
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

        self.ga_head = TralaleroTralala(algebra, in_features=self._n_mv, hidden_dim=hidden_dim)

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


def hidden_state_to_feature_map(hidden_state: torch.Tensor) -> torch.Tensor:
    if hidden_state.ndim != 3:
        raise ValueError(
            f"Expected hidden_state shape [B, N, C], got {tuple(hidden_state.shape)}"
        )

    tokens = hidden_state[:, 1:]
    batch_size, n_tokens, hidden_size = tokens.shape
    spatial_size = int(n_tokens ** 0.5)

    if spatial_size * spatial_size != n_tokens:
        raise ValueError(f"Expected square number of patch tokens, got {n_tokens}")

    return tokens.transpose(1, 2).reshape(
        batch_size, hidden_size, spatial_size, spatial_size
    )


def _config_get(config, field_name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _infer_hidden_size_from_config(config):
    hidden_size = _config_get(config, "hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    backbone_config = _config_get(config, "backbone_config")
    hidden_size = _config_get(backbone_config, "hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    return None


class ViTHiddenStatesBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        layers: tuple = (-1, -3, -6, -9),
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import ViTModel

        self.layers = tuple(layers)
        self.model = ViTModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.model.config.output_hidden_states = True

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.hidden_size = _infer_hidden_size_from_config(self.model.config)
        self.output_dim = (
            self.hidden_size * len(self.layers)
            if self.hidden_size is not None
            else None
        )

    def _extract_hidden_states(self, x: torch.Tensor):
        grad_enabled = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

        with torch.set_grad_enabled(grad_enabled):
            out = self.model(pixel_values=x, output_hidden_states=True)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("ViTModel did not return hidden_states")
        return hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self._extract_hidden_states(x)
        feature_maps = []
        for layer_idx in self.layers:
            try:
                hidden_state = hidden_states[layer_idx]
            except IndexError as exc:
                raise IndexError(
                    f"Requested hidden-state layer {layer_idx}, but backbone returned "
                    f"{len(hidden_states)} hidden states"
                ) from exc
            feature_maps.append(hidden_state_to_feature_map(hidden_state))

        return torch.cat(feature_maps, dim=1)


class DepthAnythingV2HiddenStatesBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        layers: tuple = (-1, -3, -6, -9),
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.layers = tuple(layers)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.model.config.output_hidden_states = True

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.hidden_size = _infer_hidden_size_from_config(self.model.config)
        self.output_dim = (
            self.hidden_size * len(self.layers)
            if self.hidden_size is not None
            else None
        )

    def _extract_hidden_states(self, x: torch.Tensor):
        grad_enabled = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

        with torch.set_grad_enabled(grad_enabled):
            out = self.model(pixel_values=x, output_hidden_states=True)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "Depth Anything V2 did not return hidden_states. "
                "Ensure the model supports output_hidden_states=True."
            )
        return hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self._extract_hidden_states(x)
        feature_maps = []
        for layer_idx in self.layers:
            try:
                hidden_state = hidden_states[layer_idx]
            except IndexError as exc:
                raise IndexError(
                    f"Requested hidden-state layer {layer_idx}, but Depth Anything V2 "
                    f"returned {len(hidden_states)} hidden states"
                ) from exc
            feature_maps.append(hidden_state_to_feature_map(hidden_state))

        return torch.cat(feature_maps, dim=1)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


def _normalize_optional_hidden_dims(hidden_dim):
    if hidden_dim is None:
        return []

    if isinstance(hidden_dim, int):
        if hidden_dim == -1:
            return []
        return [hidden_dim]

    hidden_dims = list(hidden_dim)

    if len(hidden_dims) == 0:
        return []

    if len(hidden_dims) == 1 and int(hidden_dims[0]) == -1:
        return []

    return [int(hd) for hd in hidden_dims]


class SimpleFullGeometricProductPoseHead(nn.Module):
    def __init__(
        self,
        algebra,
        input_dim: int = 512,
        input_features: int = 196,
        hidden_dim: list[int] | int | None = [32],
        out_features: int = 9,
        readout_type: str = "linear",
    ):
        super().__init__()

        if algebra is None:
            raise ValueError("algebra must be provided")

        self.algebra = algebra
        self.input_dim = int(input_dim)
        self.input_features = int(input_features)
        self.mv_dim = 2 ** algebra.dim
        self.out_features = int(out_features)
        self.readout_type = str(readout_type).lower()

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.input_features <= 0:
            raise ValueError("input_features must be positive")
        valid_readout_types = {"scalar", "mean", "linear", "grade", "rotor"}
        if self.readout_type not in valid_readout_types:
            raise ValueError(
                "readout_type must be one of "
                f"{sorted(valid_readout_types)}, got {readout_type!r}"
            )

        if self.readout_type == "rotor":
            if algebra.dim != 3:
                raise ValueError("vit_ga_readout_type='rotor' requires algebra_dim=3 / Cl(3,0)")
            if self.mv_dim != 8:
                raise ValueError("rotor readout requires mv_dim=8 for Cl(3,0)")
        elif self.out_features != 9:
            raise ValueError("out_features must be 9 to form a 3x3 pose matrix")

        hidden_dims = _normalize_optional_hidden_dims(hidden_dim)

        if any(hd <= 0 for hd in hidden_dims):
            raise ValueError(
                "all hidden_dim values must be positive, "
                "or use -1/None/[] to disable hidden layers"
            )

        self.to_mv = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.mv_dim),
        )

        self.blocks = nn.ModuleList()
        prev_features = self.input_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "gp": FullyConnectedSteerableGeometricProductLayer(
                        algebra,
                        in_features=prev_features,
                        out_features=hd,
                    ),
                    "act": MVSiLU(algebra, hd),
                })
            )
            prev_features = hd

        effective_out_features = 1 if self.readout_type == "rotor" else self.out_features
        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra,
            in_features=prev_features,
            out_features=effective_out_features,
        )

        if self.readout_type == "linear":
            self.mv_to_scalar = nn.Linear(self.mv_dim, 1)
        elif self.readout_type == "grade":
            self.grade_indices = []
            for grade in range(algebra.dim + 1):
                indices = [
                    idx for idx in range(self.mv_dim)
                    if int(idx).bit_count() == grade
                ]
                self.grade_indices.append(indices)
            self.mv_to_scalar = nn.Linear(algebra.dim + 1, 1)
        else:
            self.mv_to_scalar = None

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != patch_embeddings.device:
            _move_unregistered_tensors_to_device(self, patch_embeddings.device)
            self._extra_tensors_device = patch_embeddings.device

        if patch_embeddings.ndim != 3:
            raise ValueError(
                f"Expected patch_embeddings shape [B, N, C], got {tuple(patch_embeddings.shape)}"
            )

        batch_size, num_patches, embedding_dim = patch_embeddings.shape

        if num_patches != self.input_features:
            raise RuntimeError(
                f"GA pooling expected {self.input_features} patch tokens from config, got {num_patches}. "
                "Set --vit_ga_input_features to match the actual token count produced by the backbone/token_mlp."
            )

        if embedding_dim != self.input_dim:
            raise RuntimeError(
                f"GA pooling expected embedding_dim={self.input_dim}, got {embedding_dim}"
            )

        x = self.to_mv(patch_embeddings)

        for block in self.blocks:
            x = block["gp"](x)
            x = block["act"](x)

        out_mv = self.out(x)

        if self.readout_type == "rotor":
            mv = out_mv[:, 0, :]
            return _mv_to_rotation_matrix(mv)

        if self.readout_type == "scalar":
            out = out_mv[:, :, 0]
        elif self.readout_type == "mean":
            out = out_mv.mean(dim=-1)
        elif self.readout_type == "linear":
            out = self.mv_to_scalar(out_mv).squeeze(-1)
        elif self.readout_type == "grade":
            grade_values = []
            for indices in self.grade_indices:
                grade_component = out_mv[:, :, indices]
                grade_value = grade_component.norm(dim=-1)
                grade_values.append(grade_value)

            grade_features = torch.stack(grade_values, dim=-1)
            out = self.mv_to_scalar(grade_features).squeeze(-1)
        else:
            raise RuntimeError("Unexpected readout_type")

        return out.view(batch_size, 3, 3)


class ViTMultiLayerPoseBaseline(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        backbone_type: str = "vit",
        layers: tuple = (-1, -3, -6, -9),
        freeze_vit: bool = True,
        pooling_type: str = "mean",
        num_transformer_layers: int = 1,
        transformer_nhead: int = 8,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        algebra=None,
        ga_input_features: int = 196,
        ga_hidden_dim: list[int] | int = [32],
        ga_readout_type: str = "linear",
    ):
        super().__init__()

        self.model_name = model_name
        self.backbone_type = backbone_type
        self.layers = tuple(layers)
        self.freeze_vit = bool(freeze_vit)
        self.pooling_type = str(pooling_type).lower()

        valid_pooling_types = {
            "mean",
            "attention",
            "transformer_attention",
            "convolution",
            "ga",
        }
        if self.pooling_type not in valid_pooling_types:
            raise ValueError(
                "pooling_type must be one of "
                f"{sorted(valid_pooling_types)}, got {pooling_type!r}"
            )

        if self.pooling_type == "transformer_attention":
            if num_transformer_layers < 1:
                raise ValueError(
                    "num_transformer_layers must be >= 1 for "
                    "transformer_attention pooling"
                )
            if transformer_nhead <= 0:
                raise ValueError("transformer_nhead must be > 0")
            if transformer_ff_dim <= 0:
                raise ValueError("transformer_ff_dim must be > 0")
            if 512 % transformer_nhead != 0:
                raise ValueError(
                    "transformer_nhead must divide the token embedding dimension 512"
                )
            if not 0 <= transformer_dropout < 1:
                raise ValueError("transformer_dropout must satisfy 0 <= dropout < 1")

        if len(self.layers) == 0:
            raise ValueError("layers must contain at least one hidden-state index")

        if self.backbone_type == "vit":
            self.backbone = ViTHiddenStatesBackbone(
                model_name=self.model_name,
                layers=self.layers,
                freeze=self.freeze_vit,
            )
        elif self.backbone_type == "depth_anything_v2":
            self.backbone = DepthAnythingV2HiddenStatesBackbone(
                model_name=self.model_name,
                layers=self.layers,
                freeze=self.freeze_vit,
            )
        else:
            raise ValueError(
                "backbone_type must be one of {'vit', 'depth_anything_v2'}, "
                f"got {self.backbone_type!r}"
            )

        token_mlp_input_dim = self.backbone.output_dim
        first_token_layer = (
            nn.Linear(token_mlp_input_dim, 1024)
            if token_mlp_input_dim is not None
            else nn.LazyLinear(1024)
        )

        self.token_mlp = nn.Sequential(
            first_token_layer,
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

        if self.pooling_type == "mean":
            self.patch_transformer = nn.Identity()
            self.pool = None
        elif self.pooling_type == "attention":
            self.patch_transformer = nn.Identity()
            self.pool = AttentionPool(512)
        elif self.pooling_type == "transformer_attention":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.patch_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers,
            )
            self.pool = AttentionPool(512)
        elif self.pooling_type == "convolution":
            self.patch_transformer = nn.Identity()
            self.pool = None
            self.conv_pose_head = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.GroupNorm(8, 256),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(128, 1, kernel_size=1),
            )
        elif self.pooling_type == "ga":
            self.patch_transformer = nn.Identity()
            self.pool = None

            if algebra is None:
                raise ValueError("pooling_type='ga' requires algebra to be passed")

            self.ga_pose_head = SimpleFullGeometricProductPoseHead(
                algebra=algebra,
                input_dim=512,
                input_features=ga_input_features,
                hidden_dim=ga_hidden_dim,
                out_features=9,
                readout_type=ga_readout_type,
            )

        if self.pooling_type not in {"convolution", "ga"}:
            self.pose_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 9),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        flattened_spatial = feature_map.flatten(2)
        patch_features = flattened_spatial.transpose(1, 2)

        patch_embeddings = self.token_mlp(patch_features)

        if self.pooling_type == "ga":
            return self.ga_pose_head(patch_embeddings)

        if self.pooling_type == "convolution":
            batch_size, num_patches, embedding_dim = patch_embeddings.shape
            spatial_size = int(num_patches ** 0.5)
            if spatial_size * spatial_size != num_patches:
                raise RuntimeError(
                    "convolution pooling requires a square number of patch tokens, "
                    f"got {num_patches} tokens"
                )
            patch_grid = patch_embeddings.transpose(1, 2).reshape(
                batch_size,
                embedding_dim,
                spatial_size,
                spatial_size,
            )
            out = self.conv_pose_head(patch_grid)
            return out.squeeze(1)

        if self.pooling_type == "mean":
            global_embedding = patch_embeddings.mean(dim=1)
        elif self.pooling_type == "attention":
            global_embedding = self.pool(patch_embeddings)
        elif self.pooling_type == "transformer_attention":
            patch_embeddings = self.patch_transformer(patch_embeddings)
            global_embedding = self.pool(patch_embeddings)
        else:
            raise RuntimeError("Unexpected pooling_type")

        out = self.pose_head(global_embedding)
        return out.view(-1, 3, 3)


class I2S_Backbone(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        mv_per_position: int = 1,
        adapter_mid_channels: int = 0,
        adapter_high_channels: int = 0,
        adapter_output_size: int = 16,
        ga_head_type: str = "tralalero",
        ga_head_mixing_layer: str = "gp",
        ga_head_num_blocks: int = 2,
        ga_head_dropout: float = 0.0,
        ga_head_use_layer_norm: bool = False,
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)
        self.backbone_name = str(backbone_name).lower()
        self.hidden_dim = hidden_dim

        self._mv_dim = int(2**algebra.dim)
        self.mv_per_position = int(mv_per_position)
        if self.mv_per_position <= 0:
            raise ValueError("mv_per_position must be positive")

        self.conv_adapter_output = int(adapter_output_size)
        if self.conv_adapter_output <= 0:
            raise ValueError("adapter_output_size must be positive")
        self._n_mv = self.conv_adapter_output ** 2 * self.mv_per_position
        if self._n_mv > 4096:
            print(
                f"Warning: I2S_Backbone uses {self._n_mv} multivector tokens "
                f"(adapter_output_size={self.conv_adapter_output}, "
                f"mv_per_position={self.mv_per_position}). "
                "This may be memory intensive."
            )
        adapter_out_channels = self.mv_per_position * self._mv_dim
        auto_mid_channels = max(64, adapter_out_channels * 2)
        auto_high_channels = max(256, auto_mid_channels * 2)

        adapter_mid_channels = int(adapter_mid_channels)
        adapter_high_channels = int(adapter_high_channels)

        if adapter_mid_channels < -1:
            raise ValueError("adapter_mid_channels must be -1, 0, or a positive integer")

        if adapter_high_channels < -1:
            raise ValueError("adapter_high_channels must be -1, 0, or a positive integer")

        self.adapter_mid_channels = (
            auto_mid_channels
            if adapter_mid_channels == 0
            else adapter_mid_channels
        )

        self.adapter_high_channels = (
            auto_high_channels
            if adapter_high_channels == 0
            else adapter_high_channels
        )

        if 0 < self.adapter_mid_channels < adapter_out_channels:
            raise ValueError(
                "adapter_mid_channels must be -1, 0, or >= mv_per_position * mv_dim "
                f"({adapter_out_channels}), got {self.adapter_mid_channels}"
            )

        if (
            self.adapter_high_channels > 0
            and self.adapter_mid_channels > 0
            and self.adapter_high_channels < self.adapter_mid_channels
        ):
            raise ValueError(
                "adapter_high_channels must be -1, 0, or >= adapter_mid_channels "
                f"when both blocks are enabled, got high={self.adapter_high_channels}, "
                f"mid={self.adapter_mid_channels}"
            )

        if output_mode not in {"auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"}:
            raise ValueError("output_mode must be one of: auto, rotation_matrix, fourier, rotor, multivector_rotor")
        self.output_mode = output_mode
        if self.output_mode in {"rotor", "multivector_rotor"} and self._mv_dim != 8:
            raise ValueError(
                "I2S_Backbone rotor/multivector_rotor modes currently require Cl(3,0), "
                f"expected mv_dim=8, got mv_dim={self._mv_dim}."
            )

        self.backbone, backbone_channels = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained_backbone=pretrained_backbone,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        adapter_layers = []
        adapter_in_channels = backbone_channels

        if self.adapter_high_channels > 0:
            adapter_layers.extend(
                [
                    nn.Conv2d(
                        adapter_in_channels,
                        self.adapter_high_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.adapter_high_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            adapter_in_channels = self.adapter_high_channels
        else:
            adapter_layers.append(nn.Identity())

        if self.adapter_mid_channels > 0:
            adapter_layers.extend(
                [
                    nn.Conv2d(
                        adapter_in_channels,
                        self.adapter_mid_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.adapter_mid_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            adapter_in_channels = self.adapter_mid_channels
        else:
            adapter_layers.append(nn.Identity())

        adapter_layers.extend(
            [
                nn.Conv2d(
                    adapter_in_channels,
                    adapter_out_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.AdaptiveAvgPool2d(
                    (self.conv_adapter_output, self.conv_adapter_output)
                ),
            ]
        )

        self.conv_adapter = nn.Sequential(*adapter_layers)

        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_positional_encoding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self._n_mv, self._mv_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.ga_head_type = str(ga_head_type).lower()
        self.ga_head_mixing_layer = str(ga_head_mixing_layer).lower()
        self.ga_head_num_blocks = int(ga_head_num_blocks)
        self.ga_head_dropout = float(ga_head_dropout)
        self.ga_head_use_layer_norm = bool(ga_head_use_layer_norm)

        if self.ga_head_num_blocks <= 0:
            raise ValueError("ga_head_num_blocks must be positive")

        if self.ga_head_dropout < 0.0 or self.ga_head_dropout >= 1.0:
            raise ValueError("ga_head_dropout must be in [0, 1)")

        self.num_coeffs = _so3_num_fourier_coeffs(self.lmax)
        self.ga_head_fourier = self._build_ga_head(out_features=self.num_coeffs)
        self.ga_head_rotation = self._build_ga_head(out_features=9)
        self.ga_head_rotor = self._build_ga_head(out_features=4)
        self.ga_head_mv_rotor = self._build_ga_head(out_features=1)

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def _build_ga_head(self, out_features: int):
        if self.ga_head_type == "tralalero":
            return TralaleroTralala(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "transformer_like":
            return TransformerLikeMVHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "reduced":
            return ReducedGeometricProductHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                mixing_layer=self.ga_head_mixing_layer,
            )
        if self.ga_head_type == "residual_gp":
            return ResidualGeometricProductHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                num_blocks=self.ga_head_num_blocks,
                dropout=self.ga_head_dropout,
                use_layer_norm=self.ga_head_use_layer_norm,
            )
        raise ValueError(
            "Unsupported ga_head_type: "
            f"{self.ga_head_type}. Expected one of: "
            "tralalero, transformer_like, reduced, residual_gp"
        )

    def _build_backbone(self, backbone_name: str, pretrained_backbone: bool):
        if backbone_name == "resnet50":
            backbone_weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            resnet = torchvision.models.resnet50(weights=backbone_weights)
            backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            return backbone, 2048

        if backbone_name == "convnext_tiny":
            backbone_weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
            convnext = torchvision.models.convnext_tiny(weights=backbone_weights)
            return convnext.features, 768

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _resolve_mode(self):
        if self.output_mode != "auto":
            return self.output_mode
        return "fourier"

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)

        b, c, h, w = adapted.shape

        expected_c = self.mv_per_position * self._mv_dim
        expected_h = self.conv_adapter_output
        expected_w = self.conv_adapter_output

        if (c, h, w) != (expected_c, expected_h, expected_w):
            raise RuntimeError(
                f"Expected adapted features [B, {expected_c}, {expected_h}, {expected_w}], "
                f"got [B, {c}, {h}, {w}]"
            )

        tokens = adapted.reshape(
            b,
            self.mv_per_position,
            self._mv_dim,
            h,
            w,
        )

        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            b,
            h * w * self.mv_per_position,
            self._mv_dim,
        )

        if tokens.shape[1] != self._n_mv or tokens.shape[2] != self._mv_dim:
            raise RuntimeError(
                f"Expected tokens [B, {self._n_mv}, {self._mv_dim}], "
                f"got {list(tokens.shape)}"
            )

        if self.use_positional_encoding:
            tokens = tokens + self.positional_embedding

        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            _move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        mode = self._resolve_mode()
        tokens = self._encode_tokens(x)
        if mode == "fourier":
            coeffs_mv = self.ga_head_fourier(tokens)
            coeffs = coeffs_mv[..., 0]
            logits = self.logits_on_grid(coeffs)
            logits = logits / max(self.temperature, 1e-8)
            return logits
        if mode == "rotation_matrix":
            rot_mv = self.ga_head_rotation(tokens)
            rot = rot_mv[..., 0].reshape(x.shape[0], 3, 3)
            return rot
        if mode == "rotor":
            rotor_mv = self.ga_head_rotor(tokens)
            rotor = rotor_mv[..., 0]
            rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return rotor
        if mode == "multivector_rotor":
            mv = self.ga_head_mv_rotor(tokens)
            mv = mv[:, 0, :]
            return mv
        raise ValueError(f"Unsupported mode: {mode}")

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
        out = self.forward(x)
        mode = self._resolve_mode()

        if mode == "fourier":
            idx = self.predict_rotmat(out)
            return self.so3_rotmats_cache[idx]

        if mode == "rotation_matrix":
            return out

        if mode == "rotor":
            return _unit_quaternion_to_matrix(out)

        if mode == "multivector_rotor":
            return _mv_to_rotation_matrix(out)

        raise ValueError(f"Unsupported mode: {mode}")

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)


I2S_ResNet = I2S_Backbone
