"""Backbone feature map -> Cl(3,0) multivectors that really rotate with the image.

The pooled-and-reshaped tokens (`ImageToMultivectors(conv_adapter=False)`) declare 8 backbone
channels to be a multivector, but nothing makes those channels transform like one when the image
rotates, so the condition head's equivariance acts on a transformation the data never undergoes.

Here the backbone channels stay scalars and rotation enters only through *where* a feature sits.
Each cell r of the (grid x grid) map, measured from the centre (the optical axis after the warp),
has a fixed in-plane direction n(r) = (u, v) / |r|. Every output token k is

    1, e3, e12, e123 :  sum_r alpha(|r|) w(r)            (invariant under rotations about e3)
    (e1, e2)         :  sum_r beta(|r|)  w(r) n(r)       (rotates as an in-plane vector)
    (e13, e23)       :  sum_r gamma(|r|) w'(r) n(r)      (rotates the same way)

with w the per-cell scalar weights of 1x1 convolutions (they see one cell only, with weights shared
by all cells, so they commute with moving cells around) and alpha, beta, gamma learned radial
profiles. Rotating the map by 90 degrees therefore rotates every token about e3 exactly, the
induced/restricted-representation constraint of Howell et al. (NeurIPS 2023) for this head. Other
angles hold only approximately on the square grid. The backbone itself stays non-equivariant.

Axes follow the pose labels' camera frame (checked on the warp pipeline): e1 = image columns
(right), e2 = image rows (down), e3 = the optical axis. `torch.rot90(x, 1, dims=(2, 3))`, a
counter-clockwise turn on screen, maps a pose label R to R_z(-90 deg) R, and the tokens turn the
same way.

The image tokens are divided by their root-mean-square norm (a rotation invariant, so this keeps
the equivariance) and scaled by a learned gain; without it they start ~60x smaller than the pooled
ResNet tokens, which stalled the first training run for 7 epochs.

A final constant e3 token tells the downstream E(3)-equivariant heads which axis is the optical one,
reducing their symmetry to rotations about it.

split_norm=True normalises the invariant slots (1, e3, e12, e123) and the directional slots
(e1, e2, e13, e23) separately, each with its own learned gain. With one joint RMS the directional
slots start ~20x smaller than the invariant ones (the n(r) sums nearly cancel), so the condition
tokens are effectively rotation-invariant at initialisation and the frame-token flow starts at a
symmetric saddle. Each group's RMS is rotation invariant, so the equivariance is unchanged.

up_token=True adds one more constant token, -e2 (the image's up direction in the label camera frame),
and so breaks that last SO(2) on purpose: Pascal3D photos are nearly always upright, and with only
image-derived in-plane vectors the model has no in-plane reference until it learns to extract one
(the 17-epoch plateau of clifford_flow_gatr_so2_frame_v2). The image tokens themselves still rotate
with the image.
"""

import torch
import torch.nn as nn


class SO2ConditionHead(nn.Module):
    """(B, c_in, grid, grid) scalar field -> (B, n_out, 8) Cl(3,0) multivectors.

    The first tokens come from the feature map, then the constant e3 axis (and with up_token the
    constant -e2 image-up direction), n_out in total.
    Blade order 1, e1, e2, e3, e12, e13, e23, e123 (the clifford package's, see gatr_denoiser).
    """

    _INVARIANT = (0, 3, 4, 7)  # 1, e3, e12, e123
    _VECTOR = (1, 2)           # e1, e2
    _BIVECTOR = (5, 6)         # e13, e23 = n ^ e3

    def __init__(self, c_in: int = 2048, n_out: int = 256, channels: int = 128, grid: int = 7,
                 up_token: bool = False, split_norm: bool = False):
        super().__init__()
        self.split_norm = bool(split_norm)
        self.up_token = bool(up_token)
        n_const = 2 if self.up_token else 1
        if n_out <= n_const:
            raise ValueError("n_out must leave room for at least one image token besides the constants")
        if grid % 2 == 0:
            raise ValueError("grid must be odd so the centre cell sits on the rotation axis")
        self.k = n_out - n_const
        self.grid = grid
        self.reduce = nn.Sequential(nn.Conv2d(c_in, channels, 1), nn.GELU())
        self.maps = nn.Conv2d(channels, 6 * self.k, 1)   # 4 invariant + vector + bivector weights

        offsets = torch.arange(grid) - grid // 2
        rows, cols = torch.meshgrid(offsets, offsets, indexing="ij")
        u, v = cols.flatten().float(), rows.flatten().float()   # e1 right, e2 down
        rad2 = (u**2 + v**2).long()
        self.register_buffer("ridx", torch.searchsorted(rad2.unique(), rad2), persistent=False)
        norm = (u**2 + v**2).sqrt().clamp(min=1).unsqueeze(-1)    # centre cell: direction 0
        self.register_buffer("direction", torch.stack([u, v], -1) / norm, persistent=False)
        n_radii = int(self.ridx.max()) + 1
        self.radial = nn.Parameter(torch.full((6 * self.k, n_radii), 1.0 / grid**2))
        # Learned output scale (a scalar, so rotation invariant). 2 ~ the norm of an 8-channel group of
        # pooled pretrained ResNet features, the tokens this head replaces.
        self.gain = nn.Parameter(torch.tensor(2.0))
        if self.split_norm:   # gain then scales the invariant group, gain_dir the directional one
            self.gain_dir = nn.Parameter(torch.tensor(2.0))

    def forward(self, fmap):
        b, _, h, w = fmap.shape
        if (h, w) != (self.grid, self.grid):
            raise ValueError(f"SO2ConditionHead expects a {self.grid}x{self.grid} map, got {h}x{w}")
        weights = self.maps(self.reduce(fmap)).flatten(2)                # (B, 6k, cells)
        weights = (weights * self.radial[:, self.ridx]).view(b, self.k, 6, -1)
        direction = self.direction.to(fmap.dtype)
        tokens = fmap.new_zeros(b, self.k, 8)
        tokens[..., list(self._INVARIANT)] = weights[:, :, :4].sum(-1)
        tokens[..., list(self._VECTOR)] = weights[:, :, 4] @ direction
        tokens[..., list(self._BIVECTOR)] = weights[:, :, 5] @ direction
        if self.split_norm:
            inv, dirn = list(self._INVARIANT), list(self._VECTOR + self._BIVECTOR)
            rms_inv = tokens[..., inv].pow(2).sum(-1).mean(-1, keepdim=True).add(1e-6).sqrt()
            rms_dir = tokens[..., dirn].pow(2).sum(-1).mean(-1, keepdim=True).add(1e-6).sqrt()
            scaled = torch.zeros_like(tokens)
            scaled[..., inv] = self.gain * tokens[..., inv] / rms_inv.unsqueeze(-1)
            scaled[..., dirn] = self.gain_dir * tokens[..., dirn] / rms_dir.unsqueeze(-1)
        else:
            rms = tokens.pow(2).sum(-1).mean(-1, keepdim=True).add(1e-6).sqrt()   # (B, 1)
            scaled = self.gain * tokens / rms.unsqueeze(-1)
        const = fmap.new_zeros(b, 2 if self.up_token else 1, 8)
        const[:, 0, 3] = 1.0                                                   # the optical axis e3
        if self.up_token:
            const[:, 1, 2] = -1.0                                              # image up: -e2 (e2 = rows, down)
        return torch.cat([scaled, const], dim=1)
