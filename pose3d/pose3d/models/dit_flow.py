"""DiTFlow: non-equivariant flow-matching baseline with a DiT-style transformer denoiser.

Same task, data, image input and flow as CliffordFlow; only the two heads are swapped for
plain transformers, so a comparison isolates the denoiser:

* image -> ResNet backbone -> ImageToMultivectors (no ConvAdapter: the pooled backbone vector
  cut into 256 groups of 8 numbers) -> the same 256 tokens CliffordFlow's condition head sees;
* condition head (once per image): per-token linear embedding + learned positions, one
  pre-LN transformer block over the 256 tokens, then a learned mix down to `n_cond_tokens`
  (64, as CliffordFlow's n_cond_mv) condition tokens;
* vector field (every flow step): a DiT (Peebles & Xie 2023) over [x_t token] + condition
  tokens, with adaLN-Zero blocks driven by the flow time and the pooled condition; the
  velocity is read from the x_t token by a zero-initialised final layer, so training starts
  from the identity flow, as CliffordFlow does;
* flow: geodesic flow matching on SO(3) rotation matrices (as MLPFlow): MSE between the
  predicted and the true body-frame velocity log(R0^T R1), `n_time_samples` (t, R0) pairs per
  image sharing one conditioning pass, Euler sampling and the geodesic medoid of
  `n_samples` draws at evaluation.
"""

import math

import torch
import torch.nn as nn

import pose3d.geometry.matrix as matrix_flow
from pose3d.models.clifford_flow import ImageToMultivectors


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1], scaled to DiT's 0..1000 timestep range. (n, dim)"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


class DiTBlock(nn.Module):
    """Self-attention + MLP, each gated and modulated by adaLN-Zero from the conditioning c."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # adaLN-Zero: every block starts as the identity.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h, h, h, need_weights=False)[0]
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class DiTConditionHead(nn.Module):
    """256 backbone tokens -> n_out condition tokens of width dim (runs once per image)."""

    def __init__(self, n_in: int, in_dim: int, n_out: int, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_in, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.block = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=int(dim * mlp_ratio), dropout=0.0,
                                                activation="gelu", batch_first=True, norm_first=True)
        self.mix = nn.Linear(n_in, n_out)   # learned mixing along the token axis
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):                      # (B, n_in, in_dim)
        x = self.block(self.embed(tokens) + self.pos)
        x = self.mix(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)                         # (B, n_out, dim)


class PointSetEncoder(nn.Module):
    """A point cloud (B, N, 3) -> n_out condition tokens of width dim, for DiT on 3D input (no symmetry built in).

    Per-point MLP on the raw coordinates, one transformer layer over the points (permutation-equivariant, no
    positional embedding: a cloud has no order), then n_out learned queries attend to the points.
    """

    def __init__(self, n_out: int, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.block = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=int(dim * mlp_ratio), dropout=0.0,
                                                activation="gelu", batch_first=True, norm_first=True)
        self.query = nn.Parameter(0.02 * torch.randn(1, n_out, dim))
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points):                      # (B, N, 3)
        x = self.block(self.embed(points))
        q = self.query.expand(points.shape[0], -1, -1)
        return self.norm(self.pool(q, x, x, need_weights=False)[0])


class SphereTokens(nn.Module):
    """The SO(3)-Fourier head's sphere signal as plain DiT tokens (no symmetry built in): ResNet map -> 1x1 conv ->
    Gaussian lift onto the 88 HEALPix hemisphere points (SphereLift, exactly the SO(3)-Fourier / GATr input),
    then one token per point: its c_s lifted features and its direction x_k, (B, P, c_s + 3)."""

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter                      # ImageToMultivectors with a SphereLift conv_adapter

    @property
    def n_tokens(self):
        return self.adapter.conv_adapter.points.shape[0]

    @property
    def width(self):
        return self.adapter.conv_adapter.c_s + 3

    def forward(self, x):
        sig = self.adapter(x)                       # (B, c_s, P)
        pts = self.adapter.conv_adapter.points.to(sig.dtype).expand(sig.shape[0], -1, -1)
        return torch.cat([sig.transpose(1, 2), pts], dim=-1)


class DiTVectorField(nn.Module):
    """DiT over [x_t token] + condition tokens; returns the body-frame velocity (n, 3)."""

    def __init__(self, dim: int, blocks: int, heads: int, mlp_ratio: float = 4.0, pose_lmax: int = 0):
        super().__init__()
        # pose_lmax = 0: the flattened rotation matrix (9 numbers). pose_lmax = L > 0: the Wigner-D matrices
        # D^l(R), l = 1..L, flattened (sum (2l+1)^2 numbers; 454 for L = 6), the harmonic pose description the
        # SO(3)-Fourier read-out uses, without any symmetry constraint (an ablation of that model's gain).
        self.pose_lmax = int(pose_lmax)
        if self.pose_lmax:
            from pose3d.models.so3fourier_head import harmonic_basis
            X, pinvs = harmonic_basis(self.pose_lmax)
            self.register_buffer("basis_points", X, persistent=False)
            for l, P in enumerate(pinvs):
                self.register_buffer(f"basis_pinv{l}", P, persistent=False)
        n_pose = 9 if not self.pose_lmax else sum((2 * l + 1) ** 2 for l in range(1, self.pose_lmax + 1))
        self.x_embed = nn.Linear(n_pose, dim)
        self.x_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.t_embed = TimestepEmbedder(dim)
        self.c_embed = nn.Linear(dim, dim)          # pooled condition, added to the adaLN input
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio) for _ in range(blocks))
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, 3)
        for layer in (self.final_modulation[-1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def pose_features(self, R):
        if not self.pose_lmax:
            return R.flatten(1)
        from pose3d.models.so3fourier_head import wigner_from_harmonics
        pinvs = [getattr(self, f"basis_pinv{l}") for l in range(self.pose_lmax + 1)]
        D = wigner_from_harmonics(R, self.basis_points, pinvs)
        return torch.cat([Dl.flatten(1) for Dl in D[1:]], dim=1)

    def forward(self, R, t, cond):                  # R (n, 3, 3), t (n,), cond (n, n_cond, dim)
        c = self.t_embed(t) + self.c_embed(cond.mean(1))
        x = torch.cat([self.x_embed(self.pose_features(R)).unsqueeze(1) + self.x_type, cond], dim=1)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm_final(x[:, 0]), shift, scale))


class DiTFlow(nn.Module):
    def __init__(self, algebra, pretrained_backbone: bool = True, encoder_type: str = "resnet101",
                 freeze_backbone: bool = False, n_time_samples: int = 8, n_cond_tokens: int = 64,
                 dim: int = 112, blocks: int = 4, heads: int = 4, mlp_ratio: float = 4.0, pose_lmax: int = 0,
                 point_input: bool = False, sphere_input: bool = False, sphere_channels: int = 32,
                 sphere_nside: int = 4):
        super().__init__()
        self.n_time_samples = max(1, int(n_time_samples))
        if sphere_input:
            # The 88 hemisphere points of the SO(3)-Fourier head (same 1x1 conv width, same lift) as DiT tokens:
            # the equivariant sphere part replaced by a plain DiT (the "high-l vs equivariance" control).
            lift = ImageToMultivectors(algebra, pretrained_backbone=pretrained_backbone, encoder_type=encoder_type,
                                       freeze_backbone=freeze_backbone, conv_adapter=False, cond_tokens="sphere",
                                       sphere=dict(c_s=sphere_channels, nside=sphere_nside, lift_only=True))
            self.adapter = SphereTokens(lift)
            self.condition_head = DiTConditionHead(self.adapter.n_tokens, self.adapter.width, n_cond_tokens, dim,
                                                   heads, mlp_ratio)
        elif point_input:
            # Point clouds (B, N, 3) instead of images (datasets/modelnet_points.py): no backbone.
            self.adapter = nn.Identity()
            self.condition_head = PointSetEncoder(n_cond_tokens, dim, heads, mlp_ratio)
        else:
            self.adapter = ImageToMultivectors(algebra, pretrained_backbone=pretrained_backbone,
                                               encoder_type=encoder_type, freeze_backbone=freeze_backbone,
                                               conv_adapter=False)
            self.condition_head = DiTConditionHead(self.adapter.n_mv, 2**algebra.dim, n_cond_tokens, dim, heads,
                                                   mlp_ratio)
        self.vector_field = DiTVectorField(dim, blocks, heads, mlp_ratio, pose_lmax=pose_lmax)

    def condition(self, x):
        return self.condition_head(self.adapter(x))

    def velocity(self, R, t, cond):
        return self.vector_field(R, t, cond)

    def forward(self, x, R, t, cls=None):
        return self.velocity(R, t, self.condition(x))

    def compute_loss(self, img, rot_gt, criterion=None, cls=None, rots_eq=None, n_eq=None):
        # One conditioning pass per image, shared by n_time_samples (t, R0) pairs (as CliffordFlow).
        # cls is accepted for the trainer's calling convention and ignored, as in CliffordFlow.
        # With rots_eq (SYMSOL's equivalent poses), each R0 flows to its nearest equivalent (see
        # DataConfig.symsol_nearest_coupling) instead of to rot_gt.
        cond = self.condition(img)
        R1 = rot_gt
        k = self.n_time_samples
        if k > 1:
            cond = cond.repeat_interleave(k, dim=0)
            R1 = R1.repeat_interleave(k, dim=0)

        n = R1.shape[0]
        R0 = matrix_flow.random_rotation_matrices(n).to(R1.device)
        t = torch.rand(n, device=R1.device)
        if rots_eq is not None:
            R1 = matrix_flow.nearest_equivalent(R0, rots_eq.repeat_interleave(k, dim=0),
                                                n_eq.repeat_interleave(k, dim=0))

        Rt = matrix_flow.geodesic_interpolate(R0, R1, t)
        target = matrix_flow.relative_log(R0, R1)
        pred = self.velocity(Rt, t, cond)
        return (pred - target).pow(2).sum(-1).mean()

    def _medoid(self, mats):
        b, k = mats.shape[:2]
        a = mats.unsqueeze(2).expand(b, k, k, 3, 3).reshape(-1, 3, 3)
        c = mats.unsqueeze(1).expand(b, k, k, 3, 3).reshape(-1, 3, 3)
        dist = matrix_flow.geodesic_distance(a, c).view(b, k, k)
        idx = dist.sum(-1).argmin(-1)
        return mats[torch.arange(b, device=mats.device), idx]

    @torch.no_grad()
    def predict(self, x, cls=None, *, n_samples: int = 1, steps: int = 20):
        b = x.shape[0]
        n_samples = max(1, int(n_samples))
        cond = self.condition(x)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, dim=0)

        n = b * n_samples
        R = matrix_flow.random_rotation_matrices(n).to(x.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=x.device)
            R = R @ matrix_flow.exp_map(dt * self.velocity(R, t, cond))

        if n_samples > 1:
            R = self._medoid(R.view(b, n_samples, 3, 3))
        return R
