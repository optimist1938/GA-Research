"""Backbone feature map -> a cloud of direction tokens, one per cell (P1 of reports/geometric_redesign.md).

The SO(2) head (models/so2_head.py) folds the whole 7x7 map into a few multivectors per token, and a
single Cl(3,0) multivector only carries angular patterns of order l <= 1: anything finer than "one
arrow" is summed away. Here nothing is folded. Every cell j of the (grid x grid) map is its own token:

    multivector part:  the fixed unit direction s_j of the cell (vector grade)      "where"
    scalar channels:   a 1x1 conv of the cell's backbone features, layer-normed       "what"

GATr then attends over the set, keyed on invariants such as <R e_i, s_j> (does a hypothesised object
axis point towards cell j); see GATrCloudField in gatr_denoiser.py. A last constant token is the
optical axis e3 (with its own flag channel), which reduces the vector field's symmetry to rotations
about it.

Directions: a wide virtual pinhole, s_j proportional to (u_j / r, v_j / r, 1) with (u_j, v_j) the cell
offset from the centre (e1 = image columns, right; e2 = rows, down; e3 = optical axis, as the pose
labels) and r = grid // 2, so the middle of each edge looks 45 deg off axis and the corners 54.7 deg.
The true camera rays of a Pascal3D crop span under 5 deg, which would make every <R e_i, s_j> nearly
the same; the report's orthographic hemisphere lift does not fit the corners of a square grid into
the unit disc. Any direction that is a fixed function of the offset turning with it works: a 90 deg
turn or a left-right flip of the map permutes the cells, moves each scalar block with its cell and
maps s_j to G s_j (G = R_z(-90 deg) or F = diag(-1, 1, 1)), exactly. The backbone itself stays
non-equivariant.

Output: one packed tensor (B, grid^2 + 1, 8 + scalars + 1): Cl(3,0) multivector (blades 1, e1, e2,
e3, e12, e13, e23, e123), then the scalar channels, then the axis flag. Packing keeps the condition a
single tensor, so the samplers' repeat_interleave and everything else downstream stay unchanged.
"""

import torch
import torch.nn as nn


class DirectionCloudHead(nn.Module):
    def __init__(self, c_in: int = 2048, scalars: int = 64, grid: int = 7):
        super().__init__()
        if grid % 2 == 0 or grid < 3:
            raise ValueError("grid must be odd (>= 3) so the centre cell sits on the optical axis")
        self.grid, self.scalars = grid, scalars
        self.proj = nn.Conv2d(c_in, scalars, 1)
        self.norm = nn.LayerNorm(scalars)
        offsets = torch.arange(grid, dtype=torch.float32) - grid // 2
        rows, cols = torch.meshgrid(offsets, offsets, indexing="ij")
        ray = torch.stack([cols.flatten() / (grid // 2), rows.flatten() / (grid // 2),
                           torch.ones(grid * grid)], -1)
        self.register_buffer("direction", ray / ray.norm(dim=-1, keepdim=True), persistent=False)

    @property
    def n_tokens(self) -> int:
        return self.grid * self.grid + 1

    @property
    def width(self) -> int:
        return 8 + self.scalars + 1

    def forward(self, fmap):
        b, _, h, w = fmap.shape
        if (h, w) != (self.grid, self.grid):
            raise ValueError(f"DirectionCloudHead expects a {self.grid}x{self.grid} map, got {h}x{w}")
        feats = self.norm(self.proj(fmap).flatten(2).transpose(1, 2))     # (B, cells, scalars)
        out = fmap.new_zeros(b, self.n_tokens, self.width)
        out[:, :-1, 1:4] = self.direction.to(fmap.dtype)
        out[:, :-1, 8:8 + self.scalars] = feats
        out[:, -1, 3] = 1.0                                                # the optical axis e3
        out[:, -1, -1] = 1.0                                               # its flag
        return out
