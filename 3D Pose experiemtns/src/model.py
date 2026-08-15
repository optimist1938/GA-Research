# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from src.image_encoders import build_encoder
from src.rotor_utils import matrix_to_rotor, rotor_to_matrix, random_rotor, embed_rotor
from src.flow_matching_utils import geodesic_interpolate, geodesic_distance, relative_log, rotor_multiply, exp_map
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
        encoder_type: str = "resnet",
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self.encoder = build_encoder(encoder_type, pretrained=pretrained_backbone)
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
        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x


def _ga_to_canonical_mv(mv_grid, mv_dim):
    if mv_grid.shape[1] == mv_dim:
        return mv_grid
    e, e123, e1, e2, e13, e23 = torch.unbind(mv_grid, dim=1)
    zeros = torch.zeros_like(e)
    return torch.stack([e, e1, e2, zeros, zeros, e13, e23, e123], dim=1)


class ImageToMultivectors(nn.Module):
    # ResNet -> HeatMap -> ConvAdapter -> n multivectors (grid x grid)
    def __init__(self, algebra, grid=16, pretrained_backbone: bool = False):
        super().__init__()
        mv_dim = 2**algebra.dim
        self.backbone = build_encoder("resnet", pretrained=pretrained_backbone)
        backbone_channels = self.backbone.output_shape[0]

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(backbone_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, mv_dim, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d((grid, grid)),
        )
        self.n_mv = grid * grid

    def forward(self, x):
        fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)
        return adapted.flatten(2).transpose(1, 2)


class CliffordFlow(nn.Module):
    def __init__(self, algebra, hidden_dim=[32], n_cond_mv=4, pretrained_backbone: bool = False,
                 n_time_samples: int = 1, adapter_grid: int = 16):
        super().__init__()
        self.algebra = algebra
        self.adapter = ImageToMultivectors(algebra, grid=adapter_grid, pretrained_backbone=pretrained_backbone)
        self.n_cond_mv = n_cond_mv
        self.n_time_samples = max(1, int(n_time_samples))
        self.condition_head = TralaleroTralala(algebra, in_features=self.adapter.n_mv, hidden_dim=hidden_dim, out_features=self.n_cond_mv)
        self.vector_field = TralaleroTralala(algebra, in_features=2 + self.n_cond_mv, hidden_dim=hidden_dim, out_features=1)

        nn.init.zeros_(self.vector_field.out.weight)
        nn.init.zeros_(self.vector_field.out.linear_left.weight)

    def condition(self, x):
        mv = self.adapter(x)
        return self.condition_head(mv)

    def velocity(self, rotor, t, cond_mv):
        rotor_mv = embed_rotor(rotor, self.algebra).unsqueeze(1)
        t_mv = self.algebra.embed(t.reshape(-1, 1), (0,)).unsqueeze(1)
        inp = torch.cat([rotor_mv, t_mv, cond_mv], dim=1)
        out = self.vector_field(inp)[:, 0]
        return self.algebra.get_grade(out, 2)

    def forward(self, x, rotor, t):
        return self.velocity(rotor, t, self.condition(x))

    def compute_loss(self, img, rot_gt, criterion=None):
        # The backbone forward dominates the step cost while the vector field is
        # small, so drawing several (t, r0) pairs per image buys that many more
        # flow-matching samples for one shared conditioning pass.
        cond_mv = self.condition(img)
        r1 = matrix_to_rotor(rot_gt)

        k = self.n_time_samples
        if k > 1:
            # Both are interleaved the same way, so index i * k + j stays paired
            # with image i.
            cond_mv = cond_mv.repeat_interleave(k, dim=0)
            r1 = r1.repeat_interleave(k, dim=0)

        n = r1.shape[0]
        r0 = random_rotor(n).to(r1.device)
        t = torch.rand(n, device=r1.device)

        rt = geodesic_interpolate(r0, r1, t, self.algebra)
        target = relative_log(r0, r1, self.algebra)
        pred = self.velocity(rt, t, cond_mv)
        # Still a per-sample mean, so the value stays comparable across k.
        return (pred - target).pow(2).sum(-1).mean()

    def _medoid(self, rotors):
        '''Pick the sample closest to all the others, per batch item.

        The flow defines a distribution over poses, so a single draw is just one
        mode -- for symmetric objects, a randomly chosen one. The medoid under
        geodesic distance approximates the dominant mode without needing a
        density estimate.

        :param rotors: (B, K, 4)
        returns : (B, 4)
        '''
        b, k, _ = rotors.shape
        a = rotors.unsqueeze(2).expand(b, k, k, 4).reshape(-1, 4)
        c = rotors.unsqueeze(1).expand(b, k, k, 4).reshape(-1, 4)
        dist = geodesic_distance(a, c, self.algebra).view(b, k, k)
        idx = dist.sum(-1).argmin(-1)
        return rotors[torch.arange(b, device=rotors.device), idx]

    @torch.no_grad()
    def predict(self, x, *, n_samples: int = 1, steps: int = 20):
        b = x.shape[0]
        n_samples = max(1, int(n_samples))

        cond_mv = self.condition(x)
        if n_samples > 1:
            cond_mv = cond_mv.repeat_interleave(n_samples, dim=0)

        n = b * n_samples
        rotor = random_rotor(n).to(x.device)

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=x.device)
            v = self.velocity(rotor, t, cond_mv)
            rotor = rotor_multiply(rotor, exp_map(dt * v), self.algebra)

        if n_samples > 1:
            rotor = self._medoid(rotor.view(b, n_samples, 4))

        return rotor_to_matrix(rotor, self.algebra)


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra, encoder_type: str = "resnet", ga_pool_hw: tuple = (28, 28),
                 pretrained_backbone: bool = False):
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
            self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
            self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])
        else:
            self._n_mv = 8
            self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            enc_channels = getattr(self.backbone, "output_shape", None)[0]
            self.projective_matrix = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.ga_head = TralaleroTralala(algebra, in_features=self._n_mv)

    def forward(self, x):
        if self._use_ga_backbone:
            pooled_x = self.pre_encode_pool(x)
            mv_grid = self.backbone(pooled_x)
            mv_grid = _ga_to_canonical_mv(mv_grid, self._mv_dim)
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
    def __init__(self, encoder_type: str = "resnet", pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
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
