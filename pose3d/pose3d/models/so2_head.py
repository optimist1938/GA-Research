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
"""

import torch
import torch.nn as nn


class SO2ConditionHead(nn.Module):
    """(B, c_in, grid, grid) scalar field -> (B, n_out, 8) Cl(3,0) multivectors.

    The first n_out - 1 tokens come from the feature map, the last is the constant e3 axis.
    Blade order 1, e1, e2, e3, e12, e13, e23, e123 (the clifford package's, see gatr_denoiser).
    """

    _INVARIANT = (0, 3, 4, 7)  # 1, e3, e12, e123
    _VECTOR = (1, 2)           # e1, e2
    _BIVECTOR = (5, 6)         # e13, e23 = n ^ e3

    def __init__(self, c_in: int = 2048, n_out: int = 256, channels: int = 128, grid: int = 7):
        super().__init__()
        if n_out < 2:
            raise ValueError("n_out must leave room for at least one image token besides the axis")
        if grid % 2 == 0:
            raise ValueError("grid must be odd so the centre cell sits on the rotation axis")
        self.k = n_out - 1
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
        rms = tokens.pow(2).sum(-1).mean(-1, keepdim=True).add(1e-6).sqrt()   # (B, 1)
        axis = fmap.new_zeros(b, 1, 8)
        axis[..., 3] = 1.0                                                     # the optical axis e3
        return torch.cat([self.gain * tokens / rms.unsqueeze(-1), axis], dim=1)
