"""C4-lifted backbone + harmonic head: image -> Cl(3,0) condition tokens, exactly C4-equivariant.

The SO(2) head (`so2_head.py`) rotates its tokens with the *feature map*, but a ResNet's map does
not rotate with the image, so the model as a whole is not equivariant. Here the backbone is lifted
to the rotation group (Howell et al. 2023's induced representation, realised by input rotation):

    F_k = rot90( B( rot90(x, -k) ), k ),   k = 0..3                       (one batched pass)

Turning the image by 90 deg turns every F_k and shifts k by one (`lift_equivariant` in the Lean
proofs), whatever the backbone B is. The Fourier transform over k splits the four views into
C4 frequencies, per cell r of the 7x7 map (`dft_shift`):

    c0 = sum_k F_k            frequency 0 (invariant), real
    c1 = sum_k (-i)^k F_k     frequency 1, complex (turns like a vector token)
    c2 = sum_k (-1)^k F_k     frequency 2, real

A Cl(3,0) token restricted to rotations about e3 carries frequencies 0 (1, e3, e12, e123) and 1
((e1, e2), (e13, e23)) only (`Cl3.sandwich_roll`); frequency 2 cannot be written into it linearly
(`freq2_to_freq1_zero`). So every term below multiplies a harmonic with powers of the cell's
in-plane direction n(r) = u + i v (frequency 1, `so2_head`'s directions) to land on 0 or 1:

    invariant slots :  c0,   Re(conj(n) W c1),   Re(conj(n)^2 W c2)
    vector / bivector: n W c0,   W c1,   conj(n) W c2

with complex weights W (the only linear maps that commute with rotations, `rot_commutant`),
learned radial profiles and a sum over cells. Rotating the image by 90 deg therefore rotates every
token by R_z(-90 deg) exactly, the label's change (so2_head.py's convention: e1 = columns,
e2 = rows, e3 = optical axis; `torch.rot90(x, 1, dims=(2, 3))` maps R to R_z(-90 deg) R).

Unlike the SO(2) head, the frequency-1 content does not depend on cancellations of a symmetric
map: c1 measures how the features change when the image turns, which carries the scene's 'up'
from the content (an upright-trained backbone answers differently to the four views). That is the
in-plane reference the equivariant model needs, without a constant up token breaking the symmetry.

A final constant e3 token marks the optical axis, as in the SO(2) head.
"""

import torch
import torch.nn as nn


def rot90(x, k):
    """Counter-clockwise on screen (torch.rot90 over the last two dims)."""
    return torch.rot90(x, k, dims=(-2, -1))


def c4_lift(backbone, x):
    """Runs `backbone` on the 4 rotated copies of the image in one batch, rotates the maps back.

    (B, 3, H, W) -> (B, 4, C, h, w) with out[:, k] = rot90(backbone(rot90(x, -k)), k).
    With train-mode BatchNorm the batch statistics are taken over all four views together; a
    turned image has the same set of views, so the statistics (and the equivariance) are unchanged.
    """
    b = x.shape[0]
    views = torch.cat([rot90(x, -k) for k in range(4)])          # k-major: views[k*b:(k+1)*b]
    fmap = backbone(views)
    return torch.stack([rot90(fmap[k * b:(k + 1) * b], k) for k in range(4)], dim=1)


class C4HarmonicHead(nn.Module):
    """(B, 4, c_in, grid, grid) lifted maps -> (B, n_out, 8) Cl(3,0) tokens; the last one is e3.

    Blade order 1, e1, e2, e3, e12, e13, e23, e123 (the clifford package's).
    """

    _INVARIANT = (0, 3, 4, 7)   # 1, e3, e12, e123: frequency 0
    _VECTOR = (1, 2)            # e1, e2: frequency 1
    _BIVECTOR = (5, 6)          # e13, e23: frequency 1, same sense as the vector

    def __init__(self, c_in: int = 2048, n_out: int = 256, channels: int = 128, grid: int = 7):
        super().__init__()
        if n_out < 2:
            raise ValueError("n_out must leave room for at least one image token besides e3")
        if grid % 2 == 0:
            raise ValueError("grid must be odd so the centre cell sits on the rotation axis")
        k = self.k = n_out - 1
        self.grid = grid
        # Pointwise in the cell and in the view, so it commutes with both rotations and view shifts.
        self.reduce = nn.Sequential(nn.Conv2d(c_in, channels, 1), nn.GELU())
        # c0 (real): 4 invariant maps + complex weights (re, im) for the vector and the bivector.
        self.from_c0 = nn.Conv2d(channels, 8 * k, 1)
        # c1, c2: complex weights W = W_re + i W_im, 6 complex outputs per token
        # (4 invariant slots, vector, bivector). No bias: a constant would be frequency 0.
        self.c1_re = nn.Conv2d(channels, 6 * k, 1, bias=False)
        self.c1_im = nn.Conv2d(channels, 6 * k, 1, bias=False)
        self.c2_re = nn.Conv2d(channels, 6 * k, 1, bias=False)
        self.c2_im = nn.Conv2d(channels, 6 * k, 1, bias=False)

        offsets = torch.arange(grid) - grid // 2
        rows, cols = torch.meshgrid(offsets, offsets, indexing="ij")
        u, v = cols.flatten().float(), rows.flatten().float()    # e1 right, e2 down
        rad2 = (u**2 + v**2).long()
        self.register_buffer("ridx", torch.searchsorted(rad2.unique(), rad2), persistent=False)
        norm = (u**2 + v**2).sqrt().clamp(min=1)                  # centre cell: direction 0
        self.register_buffer("n_re", u / norm, persistent=False)
        self.register_buffer("n_im", v / norm, persistent=False)
        n_radii = int(self.ridx.max()) + 1
        # Radial profiles, one per (token, term): 4 + 2 (c0), 6 (c1), 6 (c2) complex-or-real terms.
        self.radial = nn.Parameter(torch.full((18 * k, n_radii), 1.0 / grid**2))
        self.gain = nn.Parameter(torch.tensor(2.0))   # as the SO(2) head: the pooled tokens' norm

    def forward(self, lifted):
        b, n_views, c, h, w = lifted.shape
        if n_views != 4 or (h, w) != (self.grid, self.grid):
            raise ValueError(f"C4HarmonicHead expects (B, 4, C, {self.grid}, {self.grid}), got {tuple(lifted.shape)}")
        f = self.reduce(lifted.flatten(0, 1)).view(b, 4, -1, h, w)
        c0 = f.sum(1)
        c1_re, c1_im = f[:, 0] - f[:, 2], f[:, 3] - f[:, 1]          # sum_k (-i)^k f_k
        c2 = f[:, 0] - f[:, 1] + f[:, 2] - f[:, 3]
        k = self.k
        cells = h * w
        rad = self.radial[:, self.ridx]                              # (18k, cells)
        r_inv0, r_vec0, r_c1, r_c2 = rad[:4 * k], rad[4 * k:6 * k], rad[6 * k:12 * k], rad[12 * k:]
        n_re, n_im = self.n_re.to(f.dtype), self.n_im.to(f.dtype)

        a0 = self.from_c0(c0).flatten(2)                              # (B, 8k, cells)
        inv = (a0[:, :4 * k] * r_inv0).sum(-1).view(b, k, 4)
        # n * (w_re + i w_im) * c0 for the vector (first k) and bivector (next k)
        w_re, w_im = a0[:, 4 * k:6 * k] * r_vec0, a0[:, 6 * k:] * r_vec0
        z0_re = (w_re * n_re - w_im * n_im).sum(-1)
        z0_im = (w_re * n_im + w_im * n_re).sum(-1)

        # W c1, complex
        p_re = (self.c1_re(c1_re) - self.c1_im(c1_im)).flatten(2) * r_c1
        p_im = (self.c1_re(c1_im) + self.c1_im(c1_re)).flatten(2) * r_c1
        inv = inv + (p_re[:, :4 * k] * n_re + p_im[:, :4 * k] * n_im).sum(-1).view(b, k, 4)   # Re(conj(n) p)
        z1_re, z1_im = p_re[:, 4 * k:].sum(-1), p_im[:, 4 * k:].sum(-1)

        # W c2 (c2 real), complex
        q_re = self.c2_re(c2).flatten(2) * r_c2
        q_im = self.c2_im(c2).flatten(2) * r_c2
        m_re, m_im = n_re * n_re - n_im * n_im, 2 * n_re * n_im      # n^2
        inv = inv + (q_re[:, :4 * k] * m_re + q_im[:, :4 * k] * m_im).sum(-1).view(b, k, 4)   # Re(conj(n)^2 q)
        z2_re = (q_re[:, 4 * k:] * n_re + q_im[:, 4 * k:] * n_im).sum(-1)                     # conj(n) q
        z2_im = (q_im[:, 4 * k:] * n_re - q_re[:, 4 * k:] * n_im).sum(-1)

        z_re, z_im = (z0_re + z1_re + z2_re).view(b, 2, k), (z0_im + z1_im + z2_im).view(b, 2, k)
        tokens = lifted.new_zeros(b, k, 8)
        tokens[..., list(self._INVARIANT)] = inv
        tokens[..., 1], tokens[..., 2] = z_re[:, 0], z_im[:, 0]
        tokens[..., 5], tokens[..., 6] = z_re[:, 1], z_im[:, 1]
        rms = tokens.pow(2).sum(-1).mean(-1, keepdim=True).add(1e-6).sqrt()      # (B, 1), invariant
        axis = lifted.new_zeros(b, 1, 8)
        axis[:, 0, 3] = 1.0
        return torch.cat([self.gain * tokens / rms.unsqueeze(-1), axis], dim=1)
