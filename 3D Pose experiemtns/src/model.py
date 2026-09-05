# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from src.image_encoders import build_encoder, is_resnet, _IMAGENET_MEAN, _IMAGENET_STD
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
    def __init__(self, algebra, grid=16, pretrained_backbone: bool = False,
                 encoder_type: str = "resnet"):
        super().__init__()
        if not is_resnet(encoder_type):
            # The conv adapter is sized from a deep CNN's channel count; the GA
            # encoders emit a handful of channels at full resolution instead.
            raise ValueError(
                f"ImageToMultivectors expects a resnet backbone, got {encoder_type!r}"
            )
        mv_dim = 2**algebra.dim
        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
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
                 n_time_samples: int = 1, adapter_grid: int = 16, encoder_type: str = "resnet"):
        super().__init__()
        self.algebra = algebra
        self.adapter = ImageToMultivectors(algebra, grid=adapter_grid,
                                           pretrained_backbone=pretrained_backbone,
                                           encoder_type=encoder_type)
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


class I2SReal(nn.Module):
    """The published Image2Sphere model, adapted to this repo's training harness.

    `I2S` above is not Image2Sphere: it average-pools the feature map before its
    GA head, discarding the spatial structure that the orthographic S2 projection
    exists to exploit. This wraps the real thing from `image2sphere.predictor` so
    a baseline can be trained under the same dataloader, schedule, and metric as
    the models it is meant to be compared against.

    Three seams need adapting:
      * upstream `compute_loss(img, cls, rot)` takes the class label second and
        returns `(loss, acc)`; the harness calls `compute_loss(img, rot, criterion)`
        and wants a scalar back.
      * upstream `forward` always takes a class tensor, even though
        `include_class_label` is off for PASCAL3D+. A zero tensor stands in, and
        `InMemoryDataset` drops `cls` anyway.
      * upstream `predict` evaluates on CPU and returns CPU rotations, because
        `eval_wigners` is a plain attribute that never follows the model to GPU.
        The harness compares against a CUDA ground truth, so the grid is held as
        a buffer here and the argmax runs on-device.
    """

    def __init__(
        self,
        encoder_type: str = "resnet101",
        pretrained_backbone: bool = True,
        lmax: int = 6,
        rec_level: int = 3,
        eval_rec_level: int = 3,
        num_classes: int = 12,
        sphere_fdim: int = 512,
        normalize_input: bool = True,
    ):
        super().__init__()
        from image2sphere.predictor import I2S as UpstreamI2S

        if not is_resnet(encoder_type):
            raise ValueError(
                f"I2SReal expects a resnet backbone, got {encoder_type!r}"
            )
        size = 101 if encoder_type == "resnet101" else 50
        encoder = f"resnet{size}" + ("_pretrained" if pretrained_backbone else "")

        self.net = UpstreamI2S(
            num_classes=num_classes,
            encoder=encoder,
            sphere_fdim=sphere_fdim,
            lmax=lmax,
            train_grid_rec_level=rec_level,
            train_grid_mode="healpix",
            eval_grid_rec_level=eval_rec_level,
            eval_use_gradient_ascent=False,
            include_class_label=False,
        )

        # Upstream leaves these as plain attributes so its own predict() can run
        # the rec_level-5 matmul on CPU. Re-registering them as buffers keeps
        # them beside the model instead.
        eval_wigners = self.net.eval_wigners
        eval_rotmats = self.net.eval_rotmats
        del self.net.eval_wigners
        del self.net.eval_rotmats
        self.net.register_buffer("eval_wigners", eval_wigners, persistent=False)
        self.net.register_buffer("eval_rotmats", eval_rotmats, persistent=False)

        # Pascal3D hands over [0, 1] images. Upstream feeds those straight to an
        # ImageNet-pretrained ResNet, but every other pretrained path in this repo
        # normalizes first (see ImageNetNormalized), so the baseline is fed the
        # same way as the models it is being compared against.
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std if self.normalize_input else img

    def _cls(self, img: torch.Tensor) -> torch.Tensor:
        # include_class_label is off, so this is only shape-compatibility.
        return torch.zeros(img.shape[0], 1, dtype=torch.long, device=img.device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        '''Returns the SO(3) Fourier coefficients, matching upstream's forward.'''
        return self.net(self._prep(img), self._cls(img))

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion=None) -> torch.Tensor:
        # criterion is ignored: upstream builds its own CrossEntropyLoss over the
        # training grid, and reproducing I2S means keeping that. So --loss and
        # --label_smoothing have no effect on this model.
        loss, _acc = self.net.compute_loss(self._prep(img), self._cls(img), rot_gt)
        return loss

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        fourier = self.forward(img)
        probs = torch.matmul(fourier, self.net.eval_wigners).squeeze(1)
        idx = probs.max(dim=1)[1]
        return self.net.eval_rotmats[idx].to(img.device)
