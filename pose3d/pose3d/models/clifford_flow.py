"""CliffordFlow: conditional flow matching on Spin(3) rotors with Clifford (GA) layers.

Pipeline: backbone -> ConvAdapter (feature map -> multivectors) -> condition head
(CGENN-style MLP) -> Clifford vector field over (rotor R_t, time, condition
multivectors) -> bivector velocity -> MSE in the tangent space. Inference is an
Euler ODE integration with an optional multi-sample geodesic medoid.

Experimental variants are all off by default and selected in `pose3d.config.Features`
and `FlowConfig`: `adapter_grid`, `adapter_channels`, `conv_adapter`, `vector_field_hidden_dim`,
`mlp_heads` and `fisher_prior`.
"""

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
from pose3d.models import fisher_prior
from pose3d.models.encoders import (
    DEPTH_ANYTHING_DEFAULT,
    build_encoder,
    freeze_encoder,
    is_dense_backbone,
)
from pose3d.models.ga_layers import TralaleroTralala


class ImageToMultivectors(nn.Module):
    """Backbone -> ConvAdapter -> a (grid x grid) sequence of multivectors."""

    def __init__(self, algebra, grid=16, pretrained_backbone: bool = True,
                 encoder_type: str = "resnet50",
                 depth_anything_model: str = DEPTH_ANYTHING_DEFAULT,
                 freeze_backbone: bool = False,
                 adapter_channels: int = 256,
                 conv_adapter: bool = True,
                 cond_tokens: str = "pooled",
                 so2_channels: int = 128):
        super().__init__()
        if cond_tokens not in ("pooled", "so2"):
            raise ValueError(f"cond_tokens must be 'pooled' or 'so2', got {cond_tokens!r}")
        if cond_tokens == "so2" and conv_adapter:
            raise ValueError("cond_tokens='so2' replaces the pooling, so it needs conv_adapter=False")
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
        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone,
                                      depth_anything_model=depth_anything_model)
        self.frozen_backbone = bool(freeze_backbone)
        if self.frozen_backbone:
            freeze_encoder(self.backbone)
        backbone_channels = self.backbone.output_shape[0]

        self.use_conv_adapter = bool(conv_adapter)
        self.cond_tokens = cond_tokens
        if cond_tokens == "so2":
            # Same token count as the pooled reshape (2048 -> 256), but built so that the vector
            # parts rotate with the image (see models/so2_head.py). Assumes a 224 input (7x7 map).
            from pose3d.models.so2_head import SO2ConditionHead
            self.n_mv = backbone_channels // mv_dim
            self.conv_adapter = SO2ConditionHead(backbone_channels, n_out=self.n_mv,
                                                 channels=so2_channels)
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

    def forward(self, x):
        if self.frozen_backbone:
            with torch.no_grad():
                fmap = self.backbone(x)
        else:
            fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)
        if self.cond_tokens == "so2":
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
                 fisher_checkpoint: str = None,
                 cond_tokens: str = "pooled",
                 so2_channels: int = 128,
                 pose_tokens: str = "rotor"):
        super().__init__()
        if pose_tokens not in ("rotor", "frame"):
            raise ValueError(f"pose_tokens must be 'rotor' or 'frame', got {pose_tokens!r}")
        if pose_tokens == "frame" and (vector_field != "gatr" or mlp_heads):
            raise ValueError("pose_tokens='frame' needs the GATr vector field (and no mlp_heads)")
        if cond_tokens != "pooled" and fisher_checkpoint:
            raise ValueError("cond_tokens='so2' and fisher_prior cannot be combined")
        if mlp_heads and fisher_checkpoint:
            raise ValueError("mlp_heads and fisher_prior cannot be combined")
        if vector_field not in ("clifford", "gatr"):
            raise ValueError(f"vector_field must be 'clifford' or 'gatr', got {vector_field!r}")
        if condition_head not in ("clifford", "gatr"):
            raise ValueError(f"condition_head must be 'clifford' or 'gatr', got {condition_head!r}")
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
                conv_adapter=conv_adapter, cond_tokens=cond_tokens, so2_channels=so2_channels)
            cond_in_features = self.adapter.n_mv

        if condition_head == "gatr":
            from pose3d.models.gatr_denoiser import GATrConditionHead
            self.condition_head = GATrConditionHead(
                cond_in_features, self.n_cond_mv, **(gatr or {}))
        else:
            self.condition_head = TralaleroTralala(
                algebra, in_features=cond_in_features, hidden_dim=hidden_dim,
                out_features=self.n_cond_mv)
        if vector_field == "gatr":
            from pose3d.models.gatr_denoiser import GATrVectorField
            self.vector_field = GATrVectorField(
                self.n_cond_mv, n_pose_tokens=3 if pose_tokens == "frame" else 1, **(gatr or {}))
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
        elif vector_field == "gatr":
            # Zero the output layer so training starts from a zero velocity field, as with the
            # Clifford MLP below.
            for p in self.vector_field.out.parameters():
                nn.init.zeros_(p)
        else:
            nn.init.zeros_(self.vector_field.out.weight)
            nn.init.zeros_(self.vector_field.out.linear_left.weight)

    def _features(self, img, cls):
        """Condition multivectors, plus the Fisher head's matrix A (None without a prior)."""
        if self.fisher_net is None:
            return self.condition_head(self.adapter(img)), None

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

    def velocity(self, rotor, t, cond_mv):
        t_mv = self.algebra.embed(t.reshape(-1, 1), (0,)).unsqueeze(1)
        if self.pose_tokens == "frame":
            return self._frame_velocity(rotor, t_mv, cond_mv)
        rotor_mv = embed_rotor(rotor, self.algebra).unsqueeze(1)
        inp = torch.cat([rotor_mv, t_mv, cond_mv], dim=1)
        out = self.vector_field(inp)[:, 0]
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
        inp = torch.cat([frame_mv, t_mv, cond_mv], dim=1)
        spatial = self.algebra.get_grade(self.vector_field(inp)[:, 0], 2)  # e12, e13, e23
        spatial_rotor = torch.cat([torch.zeros_like(spatial[..., :1]), spatial], dim=-1)
        body = rotor_multiply(rotor_multiply(rotor_conjugate(rotor, self.algebra), spatial_rotor,
                                             self.algebra), rotor, self.algebra)
        return body[..., 1:]

    def forward(self, x, rotor, t, cls=None):
        return self.velocity(rotor, t, self.condition(x, cls))

    def compute_loss(self, img, rot_gt, criterion=None, cls=None):
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
