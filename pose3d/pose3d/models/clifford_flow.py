"""CliffordFlow: conditional flow matching on Spin(3) rotors with Clifford (GA) layers.

Pipeline: backbone -> ConvAdapter (feature map -> multivectors) -> condition head
(CGENN-style MLP) -> Clifford vector field over (rotor R_t, time, condition
multivectors) -> bivector velocity -> MSE in the tangent space. Inference is an
Euler ODE integration with an optional multi-sample geodesic medoid.

Experimental variants are all off by default and selected in `pose3d.config.Features`
and `FlowConfig`: `adapter_grid`, `adapter_channels`, `conv_adapter`, `vector_field_hidden_dim`,
`mlp_heads`, `fisher_prior` and `stf_track` (the two-tower model, see `StfCondition` below and
pose3d/models/stf_track.py).
"""

from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

from pose3d.geometry.flow import (
    exp_map,
    geodesic_distance,
    geodesic_interpolate,
    relative_log,
    rotor_conjugate,
    rotor_multiply,
)
from pose3d.geometry.rotor import embed_rotor, matrix_to_rotor, random_rotor, rotor_to_matrix
from pose3d.geometry.matrix import nearest_equivalent
from pose3d.models import fisher_prior
from pose3d.models.encoders import (
    DEPTH_ANYTHING_DEFAULT,
    build_encoder,
    freeze_encoder,
    is_dense_backbone,
)
from pose3d.models.ga_layers import TralaleroTralala
from pose3d.models.stf_track import StfTower, stf_feature_blocks, stf_invariants


class StfCondition(NamedTuple):
    """The condition of the two-tower model (`stf_track`): what the condition side hands the vector field.

    `mv`: the (B, n_cond_mv, 8) condition multivectors, with the STF tower's feedback already added
    to their grade-1 part. `stf`: the symmetric traceless tensors of ranks 2..stf_lmax, `stf[i]` of
    shape (B, stf_channels, 3, ..., 3) with i + 2 index axes. Without `stf_track` the condition is
    the plain multivector tensor, as before.
    """

    mv: torch.Tensor
    stf: Tuple[torch.Tensor, ...]

    def repeat_interleave(self, repeats, dim=0):
        """Same call as torch.Tensor.repeat_interleave, so code that repeats a condition per sample
        (the time-sample batching, the multi-sample predict, the SYMSOL likelihood) needs no change."""
        return StfCondition(self.mv.repeat_interleave(repeats, dim=dim),
                            tuple(s.repeat_interleave(repeats, dim=dim) for s in self.stf))


def _native_bf16(device):
    """CPU, or a GPU from sm_80 (Ampere) on; older GPUs such as the T4 only emulate bf16."""
    if device.type == "cuda":
        return torch.cuda.get_device_capability(device) >= (8, 0)
    return device.type == "cpu"


class ImageToMultivectors(nn.Module):
    """Backbone -> ConvAdapter -> a (grid x grid) sequence of multivectors."""

    def __init__(self, algebra, grid=16, pretrained_backbone: bool = True,
                 encoder_type: str = "resnet50",
                 depth_anything_model: str = DEPTH_ANYTHING_DEFAULT,
                 freeze_backbone: bool = False,
                 adapter_channels: int = 256,
                 conv_adapter: bool = True,
                 bf16: bool = False,
                 cond_tokens: str = "pooled",
                 so2_channels: int = 128,
                 so2_up_token: bool = False,
                 so2_split_norm: bool = False,
                 so2_tokens: int = 0,
                 dircloud_scalars: int = 64,
                 dircloud_up_token: bool = False,
                 sphere: dict = None,
                 points: int = 0):
        super().__init__()
        if cond_tokens not in ("pooled", "so2", "dircloud", "spherecloud", "sphere", "points", "pointsphere",
                               "pointcloud", "pointgatrsphere", "spheregatr"):
            raise ValueError(f"cond_tokens must be 'pooled', 'so2', 'dircloud', 'spherecloud', 'sphere', 'points' "
                             f"or 'pointsphere', got {cond_tokens!r}")
        if cond_tokens != "pooled" and conv_adapter:
            raise ValueError(f"cond_tokens={cond_tokens!r} replaces the pooling, so it needs conv_adapter=False")
        if not is_dense_backbone(encoder_type):
            # The conv adapter is sized from a deep backbone's channel count; the
            # GA encoders emit a handful of channels at full resolution instead.
            raise ValueError(
                f"ImageToMultivectors expects a dense backbone, got {encoder_type!r}"
            )
        if grid < 1:
            raise ValueError("grid must be a positive integer")
        if adapter_channels < 1:
            raise ValueError("adapter_channels must be a positive integer")

        mv_dim = 2**algebra.dim
        if cond_tokens in ("points", "pointsphere", "pointcloud", "pointgatrsphere"):
            # The input is a point cloud (B, N, 3), not an image: no backbone (datasets/modelnet_points.py).
            self.backbone, self.frozen_backbone, self.use_conv_adapter = nn.Identity(), False, False
            self.bf16 = False   # nothing to run in bf16 without a backbone
            self.cond_tokens = cond_tokens
            if cond_tokens == "pointcloud":
                # The raw cloud, for the GATr condition head (PointGATrEncoder) to compress.
                self.conv_adapter, self.n_mv = nn.Identity(), 0
            elif cond_tokens == "points":
                from pose3d.models.dircloud_head import PointCloudTokens
                self.conv_adapter = PointCloudTokens(points or 256)
                self.n_mv = self.conv_adapter.n_tokens
            elif cond_tokens == "pointgatrsphere":
                from pose3d.models.so3fourier_head import PointGATrSphereHead
                sph = dict(sphere or {})
                enc = sph.pop("gatr", {})
                self.conv_adapter = PointGATrSphereHead(**{k: v for k, v in sph.items() if k != "nside"}, **enc)
                self.n_mv = self.conv_adapter.c_f
            else:
                from pose3d.models.so3fourier_head import PointSphereHead
                self.conv_adapter = PointSphereHead(**{k: v for k, v in (sphere or {}).items() if k != "nside"})
                self.n_mv = self.conv_adapter.c_f
            return
        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone,
                                      depth_anything_model=depth_anything_model)
        self.frozen_backbone = bool(freeze_backbone)
        if self.frozen_backbone:
            freeze_encoder(self.backbone)
        # The backbone alone under bf16 autocast, channels_last; the GA heads stay fp32 (lgatr's
        # advice: mixed precision hurts the multivector maths).
        self.bf16 = bool(bf16)
        if self.bf16:
            self.backbone.to(memory_format=torch.channels_last)
        backbone_channels = self.backbone.output_shape[0]

        self.use_conv_adapter = bool(conv_adapter)
        self.cond_tokens = cond_tokens
        if cond_tokens == "so2":
            # Same token count as the pooled reshape (2048 -> 256), but built so that the vector
            # parts rotate with the image (see models/so2_head.py). Assumes a 224 input (7x7 map).
            from pose3d.models.so2_head import SO2ConditionHead
            self.n_mv = so2_tokens or backbone_channels // mv_dim
            self.conv_adapter = SO2ConditionHead(backbone_channels, n_out=self.n_mv,
                                                 channels=so2_channels, up_token=so2_up_token,
                                                 split_norm=so2_split_norm)
            return
        if cond_tokens == "dircloud":
            # One token per cell of the 7x7 map (direction + scalar features) plus the e3 axis,
            # packed (B, 50, 8 + dircloud_scalars + 1); see models/dircloud_head.py.
            from pose3d.models.dircloud_head import DirectionCloudHead
            self.conv_adapter = DirectionCloudHead(backbone_channels, scalars=dircloud_scalars,
                                                   up_token=dircloud_up_token)
            self.n_mv = self.conv_adapter.n_tokens
            return
        if cond_tokens == "spherecloud":
            # P1's cloud from the image2sphere lift: the 88 hemisphere points and their lifted signal, plus the
            # e3 axis, packed (B, 89, 8 + dircloud_scalars + 1); see models/dircloud_head.py.
            from pose3d.models.dircloud_head import SphereCloudHead
            self.conv_adapter = SphereCloudHead(backbone_channels, scalars=dircloud_scalars)
            self.n_mv = self.conv_adapter.n_tokens
            return
        if cond_tokens == "spheregatr":
            # The SO(3)-Fourier lift with a GATr encoder over the hemisphere points (models/so3fourier_head.py).
            from pose3d.models.so3fourier_head import SphereGATrHead
            sph = dict(sphere or {})
            enc = sph.pop("gatr", {})
            self.conv_adapter = SphereGATrHead(backbone_channels, **sph, **enc)
            self.n_mv = self.conv_adapter.c_f
            return
        if cond_tokens == "sphere":
            sphere = dict(sphere or {})
            if sphere.pop("lift_only", False):
                # Scalar signals at the hemisphere points, (B, c_s, P); models/rotor_corr_head.py reads them.
                from pose3d.models.so3fourier_head import SphereLift
                self.conv_adapter = SphereLift(backbone_channels, **{k: sphere[k] for k in ("c_s", "nside")
                                                                     if k in sphere})
                self.n_mv = self.conv_adapter.c_s
                return
            # Wigner coefficients of signals on SO(3), (B, c_f, sum_l (2l+1)^2); models/so3fourier_head.py.
            from pose3d.models.so3fourier_head import SphereConditionHead
            self.conv_adapter = SphereConditionHead(backbone_channels, **sphere)
            self.n_mv = self.conv_adapter.c_f
            return
        if not self.use_conv_adapter:
            # No adapter: the globally pooled backbone vector is cut into consecutive
            # groups of mv_dim channels, one multivector each (2048 -> 256 for ResNet-50/101),
            # leaving all mixing to the condition head. grid and adapter_channels are unused.
            if backbone_channels % mv_dim:
                raise ValueError(f"{backbone_channels} backbone channels do not split into "
                                 f"{mv_dim}-component multivectors")
            self.mv_dim = mv_dim
            self.conv_adapter = nn.AdaptiveAvgPool2d(1)
            self.n_mv = backbone_channels // mv_dim
            return

        # adapter_channels sizes the first 1x1 conv (backbone_channels -> adapter_channels),
        # by far the biggest matrix in the non-backbone model at the default 256. The
        # second conv keeps the 4x bottleneck ratio of the original 256 -> 64 layout.
        mid_channels = max(8, adapter_channels // 4)
        self.conv_adapter = nn.Sequential(
            nn.Conv2d(backbone_channels, adapter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_channels), nn.SiLU(inplace=True),
            nn.Conv2d(adapter_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mv_dim, kernel_size=1, bias=True),
            nn.AdaptiveAvgPool2d((grid, grid)),
        )
        self.n_mv = grid * grid

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def _backbone(self, x):
        if not (self.bf16 and _native_bf16(x.device)):
            return self.backbone(x)
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            return self.backbone(x.contiguous(memory_format=torch.channels_last)).float()

    def forward(self, x):
        if self.frozen_backbone:
            with torch.no_grad():
                fmap = self._backbone(x)
        else:
            fmap = self._backbone(x)
        adapted = self.conv_adapter(fmap)
        if self.cond_tokens in ("so2", "dircloud", "spherecloud", "sphere", "points", "pointsphere", "pointcloud",
                                "pointgatrsphere", "spheregatr"):
            return adapted
        if not self.use_conv_adapter:
            return adapted.flatten(1).view(adapted.shape[0], self.n_mv, self.mv_dim)
        return adapted.flatten(2).transpose(1, 2)


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


class MultivectorMLP(nn.Module):
    """Plain MLP standing in for TralaleroTralala in the `mlp_heads` ablation.

    Takes and returns (B, n_mv, mv_dim) like the geometric-algebra head, so it is
    a drop-in swap that flattens the multivectors and ignores their algebraic
    structure. One hidden layer per entry of hidden_dim, all of a single width
    chosen so the total parameter count lands as close to target_params as
    possible.
    """

    def __init__(self, in_mv: int, out_mv: int, mv_dim: int, n_hidden: int, target_params: int):
        super().__init__()
        self.out_mv, self.mv_dim = out_mv, mv_dim
        in_f, out_f = in_mv * mv_dim, out_mv * mv_dim

        def dims(width):
            return [in_f] + [width] * n_hidden + [out_f]

        def count(width):
            d = dims(width)
            return sum(a * b + b for a, b in zip(d[:-1], d[1:]))

        # Counted analytically: building a module per candidate width is slow.
        self.width = min(range(1, 8192), key=lambda w: abs(count(w) - target_params))
        d = dims(self.width)
        layers = []
        for a, b in zip(d[:-1], d[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x.flatten(1)).view(x.shape[0], self.out_mv, self.mv_dim)


def _n_hidden_layers(hidden_dim):
    return 1 if isinstance(hidden_dim, int) else len(hidden_dim)


class CliffordFlow(nn.Module):
    def __init__(self, algebra, hidden_dim=(32,), n_cond_mv: int = 64,
                 pretrained_backbone: bool = True, n_time_samples: int = 8,
                 adapter_grid: int = 16, adapter_channels: int = 256,
                 encoder_type: str = "resnet50",
                 depth_anything_model: str = DEPTH_ANYTHING_DEFAULT,
                 freeze_backbone: bool = False,
                 vector_field_hidden_dim=None,
                 conv_adapter: bool = True,
                 mlp_heads: bool = False,
                 vector_field: str = "clifford",
                 condition_head: str = "clifford",
                 gatr: dict = None,
                 gatr_zero_readout: bool = True,
                 gatr_final_norm: bool = False,
                 fisher_checkpoint: str = None,
                 backbone_bf16: bool = False,
                 cond_tokens: str = "pooled",
                 so2_channels: int = 128,
                 so2_up_token: bool = False,
                 so2_split_norm: bool = False,
                 pose_tokens: str = "rotor",
                 dircloud_scalars: int = 64,
                 dircloud_up_token: bool = False,
                 so3fourier: dict = None,
                 rotorcorr: dict = None,
                 n_points: int = 256,
                 gatr_pose_harmonics: int = 0,
                 gatr_encoder: dict = None,
                 stf_track: bool = False,
                 stf_lmax: int = 3,
                 stf_channels: int = 8,
                 stf_inject: str = "gram"):
        super().__init__()
        if stf_track and (vector_field != "gatr" or mlp_heads or fisher_checkpoint or cond_tokens != "pooled"
                          or pose_tokens != "rotor"):
            raise ValueError("stf_track needs vector_field='gatr', cond_tokens='pooled', pose_tokens='rotor' "
                             "and no mlp_heads / fisher_prior (the STF tensors are built from the pooled "
                             "condition multivectors and injected at the single rotor token)")
        # The speed options (dense_linear, compile) exist for the two modules of the reference recipe
        # (GATrVectorField, GATrConditionHead); the other GATr modules of the SYMSOL stack take the sizes only.
        gatr_plain = {k: v for k, v in (gatr or {}).items() if k not in ("dense_linear", "compile")}
        so3f = dict(so3fourier or {})
        rc = dict(rotorcorr or {})
        sphere_fields = ("so3fourier", "rotorcorr")
        if (cond_tokens in ("sphere", "pointsphere", "pointgatrsphere", "spheregatr")) != (vector_field in sphere_fields):
            raise ValueError("cond_tokens='sphere' / 'pointsphere' go with vector_field='so3fourier' or 'rotorcorr'")
        if cond_tokens in ("pointsphere", "pointgatrsphere", "spheregatr") and vector_field != "so3fourier":
            raise ValueError("cond_tokens='pointsphere' feeds the SO(3)-Fourier read-out: vector_field='so3fourier'")
        if vector_field in sphere_fields and (condition_head != "none" or pose_tokens != "rotor"
                                              or mlp_heads or fisher_checkpoint):
            raise ValueError(f"vector_field={vector_field!r} needs condition_head='none', pose_tokens='rotor', "
                             "no mlp_heads and no fisher_prior")
        if pose_tokens not in ("rotor", "frame", "frame_ch", "frame_pin"):
            raise ValueError(f"pose_tokens must be 'rotor', 'frame', 'frame_ch' or 'frame_pin', got {pose_tokens!r}")
        clouds = ("dircloud", "spherecloud", "points")
        if (cond_tokens in clouds + ("pointcloud",)) != (pose_tokens == "frame_pin"):
            raise ValueError("cond_tokens='dircloud' / 'spherecloud' / 'points' / 'pointcloud' and "
                             "pose_tokens='frame_pin' go together (P1)")
        if cond_tokens == "pointcloud" and (condition_head != "gatr" or vector_field != "gatr"):
            raise ValueError("cond_tokens='pointcloud' is the 8.95 recipe on points: condition_head='gatr' "
                             "(PointGATrEncoder) and vector_field='gatr'")
        if cond_tokens in clouds and condition_head != "none":
            raise ValueError(f"cond_tokens={cond_tokens!r} feeds the cloud straight to GATr: it needs "
                             "condition_head='none'")
        if pose_tokens != "rotor" and (vector_field != "gatr" or mlp_heads):
            raise ValueError(f"pose_tokens={pose_tokens!r} needs the GATr vector field (and no mlp_heads)")
        if cond_tokens != "pooled" and fisher_checkpoint:
            raise ValueError("cond_tokens='so2' and fisher_prior cannot be combined")
        if mlp_heads and fisher_checkpoint:
            raise ValueError("mlp_heads and fisher_prior cannot be combined")
        if vector_field not in ("clifford", "gatr", "so3fourier", "rotorcorr"):
            raise ValueError(f"vector_field must be 'clifford', 'gatr', 'so3fourier' or 'rotorcorr', "
                             f"got {vector_field!r}")
        if condition_head not in ("clifford", "gatr", "none"):
            raise ValueError(f"condition_head must be 'clifford', 'gatr' or 'none', got {condition_head!r}")
        if condition_head == "none" and (cond_tokens == "pooled" or mlp_heads):
            raise ValueError("condition_head='none' feeds the so2 / dircloud tokens straight to the vector "
                             "field: it needs cond_tokens='so2' or 'dircloud' (and no mlp_heads)")
        if "gatr" in (vector_field, condition_head) and mlp_heads:
            raise ValueError("GATr heads and mlp_heads cannot be combined")
        if condition_head == "gatr" and fisher_checkpoint:
            raise ValueError("condition_head='gatr' and fisher_prior cannot be combined")
        self.algebra = algebra
        self.n_cond_mv = n_cond_mv
        self.n_time_samples = max(1, int(n_time_samples))
        self.mlp_heads = mlp_heads
        self.pose_tokens = pose_tokens
        mv_dim = int(2**algebra.dim)
        vf_hidden_dim = hidden_dim if vector_field_hidden_dim is None else vector_field_hidden_dim

        # With fisher_checkpoint set, one shared ResNet-101 feeds cond_mv (gradient
        # from the flow-matching MSE only) and, on a detached copy, the Fisher head
        # (gradient from the Fisher NLL only). Detaching there keeps the head
        # tracking the backbone's drifting features instead of staying calibrated to
        # its pretrained-time values.
        self.fisher_net = None
        self.adapter = None
        if fisher_checkpoint:
            base = fisher_prior.resnet101()
            self.fisher_net = fisher_prior.ResnetHead(
                base, n_classes=13, embedding_dim=32, num_hidden_nodes=512, n_out=9)
            self.fisher_net.load_state_dict(torch.load(fisher_checkpoint, map_location="cpu"))

            self._fisher_n_mv = 8
            self.fisher_proj = nn.Linear(base.output_size, self._fisher_n_mv * mv_dim)
            cond_in_features = self._fisher_n_mv
        else:
            self.adapter = ImageToMultivectors(
                algebra, grid=adapter_grid, pretrained_backbone=pretrained_backbone,
                encoder_type=encoder_type, depth_anything_model=depth_anything_model,
                freeze_backbone=freeze_backbone, adapter_channels=adapter_channels,
                conv_adapter=conv_adapter, bf16=backbone_bf16, cond_tokens=cond_tokens, so2_channels=so2_channels,
                so2_up_token=so2_up_token, so2_split_norm=so2_split_norm,
                so2_tokens=n_cond_mv if condition_head == "none" else 0,
                dircloud_scalars=dircloud_scalars, dircloud_up_token=dircloud_up_token,
                sphere=(dict({k: rc[k] for k in ("c_s", "nside") if k in rc}, lift_only=True)
                        if vector_field == "rotorcorr" else
                        dict({k: so3f[k] for k in ("c_s", "c_f", "lmax", "nside") if k in so3f},
                             **({"gatr": dict(gatr_encoder or gatr_plain)}
                                if cond_tokens in ("pointgatrsphere", "spheregatr") else {}))),
                points=n_points)
            cond_in_features = self.adapter.n_mv
            if cond_tokens in clouds:
                self.n_cond_mv = self.adapter.n_mv   # the cloud's size, not a free choice

        if condition_head == "none":
            # The so2 head already emits n_cond_mv honest multivectors (incl. the e3 axis token).
            self.condition_head = nn.Identity()
        elif condition_head == "gatr" and cond_tokens == "pointcloud":
            from pose3d.models.gatr_denoiser import PointGATrEncoder
            self.condition_head = PointGATrEncoder(self.n_cond_mv, **gatr_plain)
        elif condition_head == "gatr":
            from pose3d.models.gatr_denoiser import GATrConditionHead
            self.condition_head = GATrConditionHead(
                cond_in_features, self.n_cond_mv, **(gatr or {}))
        else:
            self.condition_head = TralaleroTralala(
                algebra, in_features=cond_in_features, hidden_dim=hidden_dim,
                out_features=self.n_cond_mv)
        self.vector_field_kind = vector_field
        if vector_field == "so3fourier":
            from pose3d.models.so3fourier_head import SO3FourierVelocity
            self.vector_field = SO3FourierVelocity(
                c_f=so3f.get("c_f", 8), lmax=so3f.get("lmax", 6), hidden=so3f.get("hidden", 44),
                flip_symmetrize=so3f.get("flip_symmetrize", True), canonicalize=so3f.get("canonicalize", True))
        elif vector_field == "rotorcorr":
            from pose3d.models.rotor_corr_head import RotorCorrelationVelocity
            self.vector_field = RotorCorrelationVelocity(
                self.adapter.conv_adapter.points, c_s=self.adapter.conv_adapter.c_s,
                **{k: rc[k] for k in ("templates", "hidden", "kappa", "flip_symmetrize") if k in rc})
        elif cond_tokens == "pointcloud":
            from pose3d.models.gatr_denoiser import GATrCloudField
            # The 8.95 recipe on points: the GATr field over [pose, n_cond_mv PGA condition tokens].
            self.vector_field = GATrCloudField(self.condition_head.out_scalars, final_norm=gatr_final_norm,
                                               cond_mv=16, pose_lmax=gatr_pose_harmonics, **gatr_plain)
        elif pose_tokens == "frame_pin":
            from pose3d.models.gatr_denoiser import GATrCloudField
            # Scalar channels per condition token: the cell features plus the constant tokens' flags.
            self.vector_field = GATrCloudField(self.adapter.conv_adapter.width - 8, final_norm=gatr_final_norm,
                                               pose_lmax=gatr_pose_harmonics, **gatr_plain)
        elif vector_field == "gatr":
            from pose3d.models.gatr_denoiser import GATrVectorField
            stf_blocks = stf_feature_blocks(stf_lmax, stf_channels, stf_inject) if stf_track else None
            self.vector_field = GATrVectorField(
                self.n_cond_mv, n_pose_tokens=3 if pose_tokens == "frame" else 1,
                pose_channels=3 if pose_tokens == "frame_ch" else 1,
                extra_scalars=(gatr_pose_harmonics + 1) ** 2 if gatr_pose_harmonics and pose_tokens == "rotor" else 0,
                stf_blocks=stf_blocks, **(gatr or {}))
        else:
            self.vector_field = TralaleroTralala(
                algebra, in_features=2 + self.n_cond_mv, hidden_dim=vf_hidden_dim, out_features=1)

        if mlp_heads:
            # Ablation: same pipeline, heads swapped for plain MLPs sized to the GA
            # heads' parameter counts. Everything else (rotor/time embedding,
            # grade-2 read-out, loss, sampler) is untouched.
            ga_cond, ga_field = _n_params(self.condition_head), _n_params(self.vector_field)
            self.condition_head = MultivectorMLP(
                cond_in_features, self.n_cond_mv, mv_dim, _n_hidden_layers(hidden_dim), ga_cond)
            self.vector_field = MultivectorMLP(
                2 + self.n_cond_mv, 1, mv_dim, _n_hidden_layers(vf_hidden_dim), ga_field)
            nn.init.zeros_(self.vector_field.net[-1].weight)
            nn.init.zeros_(self.vector_field.net[-1].bias)
        elif vector_field in sphere_fields:
            pass   # SO3FourierVelocity / RotorCorrelationVelocity zero-initialise their own read-outs
        elif vector_field == "gatr" and gatr_zero_readout:
            # Zero the output layer so training starts from a zero velocity field, as with the
            # Clifford MLP below (see FlowConfig.gatr_zero_readout for when this stalls P1).
            for p in self.vector_field.out.parameters():
                nn.init.zeros_(p)
        elif vector_field == "gatr":
            pass   # GATr's own initialisation of the read-out
        else:
            nn.init.zeros_(self.vector_field.out.weight)
            nn.init.zeros_(self.vector_field.out.linear_left.weight)

        # The STF tower is built last so that, without stf_track, no parameter and no random draw
        # of the reference model changes.
        self.stf_tower = None
        self.stf_inject = stf_inject
        if stf_track:
            # Vectors in: the grade-1 parts of the n_cond_mv condition multivectors. Invariant
            # scalars for the gates: their grade-0 and grade-3 parts.
            self.stf_tower = StfTower(self.n_cond_mv, lmax=stf_lmax, channels=stf_channels,
                                      n_scalars=2 * self.n_cond_mv)

    def _couple(self, cond_mv):
        """The two-tower step on the condition side (identity without stf_track).

        The grade-1 parts of the condition multivectors (vectors, l = 1) go into the STF tower, which
        builds the l = 2..stf_lmax tensors from them; the tower's feedback vectors are added back
        onto the grade-1 parts (CliffordSTF's downward cross-track coupling). Returns an
        `StfCondition`.
        """
        if self.stf_tower is None:
            return cond_mv
        invariants = torch.cat([cond_mv[..., 0], cond_mv[..., 7]], dim=-1)   # grades 0 and 3
        out = self.stf_tower(cond_mv[..., 1:4], invariants)
        mv = torch.cat([cond_mv[..., :1], cond_mv[..., 1:4] + out.feedback, cond_mv[..., 4:]], dim=-1)
        return StfCondition(mv, out.stf)

    def _features(self, img, cls):
        """Condition multivectors, plus the Fisher head's matrix A (None without a prior)."""
        if self.fisher_net is None:
            return self._couple(self.condition_head(self.adapter(img))), None

        latent = self.fisher_net.base(img)
        mv = self.fisher_proj(latent).reshape(-1, self._fisher_n_mv, 2**self.algebra.dim)
        cond_mv = self.condition_head(mv)

        class_feat = self.fisher_net.class_embedding(cls.view(-1) + 1)
        head_in = torch.cat([latent.detach(), class_feat], dim=1)
        A = self.fisher_net.head(head_in).view(-1, 3, 3)
        return cond_mv, A

    def condition(self, x, cls=None):
        cond_mv, _ = self._features(x, cls)
        return cond_mv

    def velocity(self, rotor, t, cond):
        """Bivector velocity at the rotor `rotor` (B, 4) and time `t` (B,) given the condition
        `cond`: the (B, n_cond_mv, 8) multivectors, or an `StfCondition` with stf_track."""
        cond_mv, stf = (cond.mv, cond.stf) if isinstance(cond, StfCondition) else (cond, None)
        if self.vector_field_kind in ("so3fourier", "rotorcorr"):
            # cond_mv: (B, c_f, N) Wigner coefficients (so3fourier) or (B, c_s, P) signals at the hemisphere
            # points (rotorcorr); the read-out returns the body velocity directly.
            return self.vector_field(rotor_to_matrix(rotor, self.algebra), t, cond_mv)
        if self.pose_tokens == "frame_pin":
            return self._pin_velocity(rotor, t, cond_mv)
        t_mv = self.algebra.embed(t.reshape(-1, 1), (0,)).unsqueeze(1)
        if self.pose_tokens in ("frame", "frame_ch"):
            return self._frame_velocity(rotor, t_mv, cond_mv)
        rotor_mv = embed_rotor(rotor, self.algebra).unsqueeze(1)
        inp = torch.cat([rotor_mv, t_mv, cond_mv], dim=1)
        if getattr(self.vector_field, "extra_scalars", 0):
            # Pose harmonics (FlowConfig.gatr_pose_harmonics): Y_lm, l <= L, of every condition token's vector part
            # in the current pose's frame, R^T v / |R^T v|, as scalars of that token (zero on the rotor and time).
            from e3nn import o3
            R = rotor_to_matrix(rotor, self.algebra)
            y = torch.einsum("bji,bnj->bni", R, cond_mv[..., 1:4])
            u = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            L = int(round(self.vector_field.extra_scalars ** 0.5)) - 1
            harm = o3.spherical_harmonics(list(range(L + 1)), u, normalize=True, normalization="component")
            extra = torch.cat([harm.new_zeros(harm.shape[0], 2, harm.shape[-1]), harm], dim=1)
        else:
            extra = None
        if stf is None:
            # The Clifford-MLP field (vector_field='clifford') takes the tokens only; GATr also takes the pose-harmonic scalars.
            out = (self.vector_field(inp) if extra is None else self.vector_field(inp, extra))[:, 0]
        else:
            # The condition tensors against the CURRENT rotor: invariants of (S_l, R_t), see
            # stf_track.stf_invariants. They enter the vector field as extra scalars of the rotor token.
            matrix = rotor_to_matrix(rotor, self.algebra)
            feats = stf_invariants(stf, matrix, self.stf_inject)
            out = self.vector_field(inp, extra, stf_features=feats)[:, 0]
        return self.algebra.get_grade(out, 2)

    def _frame_velocity(self, rotor, t_mv, cond_mv):
        """Pose in as the frame R e1, R e2, R e3, velocity out in the camera frame.

        GATr acts on every token by the sandwich g x g~. On a rotor token that is conjugation,
        R -> G R G^T, while a camera rotation acts on the pose as R -> G R; with condition tokens
        that really rotate (cond_tokens='so2') the two would disagree. The columns of R are vectors
        that do go to G R e_i, and the spatial velocity w = r v r~ goes to G w, so the network's
        symmetry matches the physical one. The body velocity v = r~ w r is returned, as before.
        """
        frame = rotor_to_matrix(rotor, self.algebra).transpose(-1, -2)   # rows: R e1, R e2, R e3
        frame_mv = self.algebra.embed(frame, (1, 2, 3))
        if self.pose_tokens == "frame_ch":
            # The whole frame as 3 channels of token 0; time and condition tokens use channel 0.
            rest = torch.cat([t_mv, cond_mv], dim=1).unsqueeze(2)                 # (B, 1 + n_cond, 1, 8)
            rest = torch.cat([rest, rest.new_zeros(*rest.shape[:2], 2, 8)], dim=2)
            inp = torch.cat([frame_mv.unsqueeze(1), rest], dim=1)                  # (B, 2 + n_cond, 3, 8)
        else:
            inp = torch.cat([frame_mv, t_mv, cond_mv], dim=1)
        spatial = self.algebra.get_grade(self.vector_field(inp)[:, 0], 2)  # e12, e13, e23
        return self._spatial_to_body(rotor, spatial)

    def _pin_velocity(self, rotor, t, cond):
        """P1: the pose as l = (R e1) e123 (axial), R e2, R e3 (polar), a bivector velocity out."""
        frame = rotor_to_matrix(rotor, self.algebra).transpose(-1, -2)   # rows: R e1, R e2, R e3
        pose = frame.new_zeros(frame.shape[0], 3, 8)
        a = frame[:, 0]
        pose[:, 0, 4], pose[:, 0, 5], pose[:, 0, 6] = a[:, 2], -a[:, 1], a[:, 0]   # e12, e13, e23
        pose[:, 1, 1:4], pose[:, 2, 1:4] = frame[:, 1], frame[:, 2]
        return self._spatial_to_body(rotor, self.vector_field(pose, t, cond, R=frame.transpose(-1, -2)))

    def _spatial_to_body(self, rotor, spatial):
        """Camera-frame velocity bivector (e12, e13, e23) -> body velocity r~ w r."""
        spatial_rotor = torch.cat([torch.zeros_like(spatial[..., :1]), spatial], dim=-1)
        body = rotor_multiply(rotor_multiply(rotor_conjugate(rotor, self.algebra), spatial_rotor,
                                             self.algebra), rotor, self.algebra)
        return body[..., 1:]

    def forward(self, x, rotor, t, cls=None):
        return self.velocity(rotor, t, self.condition(x, cls))

    def compute_loss(self, img, rot_gt, criterion=None, cls=None, rots_eq=None, n_eq=None):
        # The backbone forward dominates the step cost while the vector field is
        # small, so drawing several (t, r0) pairs per image buys that many more
        # flow-matching samples for one shared conditioning pass.
        cond_mv, fisher_a = self._features(img, cls)
        r1 = matrix_to_rotor(rot_gt)

        fisher_loss = None
        if fisher_a is not None:
            fisher_loss = -fisher_prior.log_prob(fisher_a, rot_gt).mean()

        k = self.n_time_samples
        if k > 1:
            # Both are interleaved the same way, so index i * k + j stays paired
            # with image i.
            cond_mv = cond_mv.repeat_interleave(k, dim=0)
            r1 = r1.repeat_interleave(k, dim=0)
            if fisher_a is not None:
                fisher_a = fisher_a.repeat_interleave(k, dim=0)

        n = r1.shape[0]
        if fisher_a is not None:
            r0 = matrix_to_rotor(fisher_prior.sample_batch(fisher_a))
        else:
            r0 = random_rotor(n).to(r1.device)
        t = torch.rand(n, device=r1.device)
        if rots_eq is not None:
            # SYMSOL: flow each r0 to its nearest equivalent pose (DataConfig.symsol_nearest_coupling).
            R1 = nearest_equivalent(rotor_to_matrix(r0, self.algebra), rots_eq.repeat_interleave(k, dim=0),
                                    n_eq.repeat_interleave(k, dim=0))
            r1 = matrix_to_rotor(R1)

        rt = geodesic_interpolate(r0, r1, t, self.algebra)
        target = relative_log(r0, r1, self.algebra)
        pred = self.velocity(rt, t, cond_mv)
        # Still a per-sample mean, so the value stays comparable across k.
        loss = (pred - target).pow(2).sum(-1).mean()
        if fisher_loss is not None:
            loss = loss + fisher_loss
        return loss

    def _medoid(self, rotors):
        """Pick the sample closest to all the others, per batch item.

        The flow defines a distribution over poses, so a single draw is just one
        mode -- for symmetric objects, a randomly chosen one. The medoid under
        geodesic distance approximates the dominant mode without needing a
        density estimate.

        :param rotors: (B, K, 4)
        returns : (B, 4)
        """
        b, k, _ = rotors.shape
        a = rotors.unsqueeze(2).expand(b, k, k, 4).reshape(-1, 4)
        c = rotors.unsqueeze(1).expand(b, k, k, 4).reshape(-1, 4)
        dist = geodesic_distance(a, c, self.algebra).view(b, k, k)
        idx = dist.sum(-1).argmin(-1)
        return rotors[torch.arange(b, device=rotors.device), idx]

    @torch.no_grad()
    def predict(self, x, cls=None, *, n_samples: int = 1, steps: int = 20):
        b = x.shape[0]
        n_samples = max(1, int(n_samples))

        cond_mv, fisher_a = self._features(x, cls)
        if n_samples > 1:
            cond_mv = cond_mv.repeat_interleave(n_samples, dim=0)
            if fisher_a is not None:
                fisher_a = fisher_a.repeat_interleave(n_samples, dim=0)

        if fisher_a is not None:
            rotor = matrix_to_rotor(fisher_prior.sample_batch(fisher_a))
        else:
            rotor = random_rotor(b * n_samples).to(x.device)

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((rotor.shape[0],), i * dt, device=x.device)
            v = self.velocity(rotor, t, cond_mv)
            rotor = rotor_multiply(rotor, exp_map(dt * v), self.algebra)

        if n_samples > 1:
            rotor = self._medoid(rotor.view(b, n_samples, 4))

        return rotor_to_matrix(rotor, self.algebra)
