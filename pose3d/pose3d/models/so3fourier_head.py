"""SO(3)-Fourier velocity field (image2sphere / Howell style) for CliffordFlow.

Design and proofs: pose3d/reports/equivariance_alt_head.md. The GATr heads carry Cl(3,0) multivectors,
i.e. angular frequencies l <= 1 per token; here the image becomes C_f signals on SO(3) with Fourier
coefficients up to l = lmax, and the velocity at a pose hypothesis R is read from those signals expressed
around R.

Condition path (once per image), SphereConditionHead:
    fmap (B, C, 7, 7) -> 1x1 conv -> C_s scalar maps
      -> D4-steerable orthographic lift onto a HEALPix hemisphere (P points, optical axis e3):
         Gaussian weights in |cell - coverage * point| with per-radius gains, so a 90 degree turn or a
         left-right flip of the map permutes cells and points alike
      -> ReLU on the grid -> S2 Fourier transform (l <= lmax) -> per-(channel, l) gains (isotropic kernel)
      -> S2 -> SO(3) convolution with Fourier-domain filters psi (image2sphere's S2 conv)
      -> norm gate per (channel, l) block (exactly equivariant; a grid ReLU is only approximately so)
    = c (B, C_f, sum_l (2l+1)^2), the Wigner coefficients of C_f signals f(g) = sum_l <c_l, D^l(g)>.
    Camera roll G acts as c_l -> D^l(G) c_l (left action).

Velocity read-out (every flow step), SO3FourierVelocity:
    c~_l = D^l(R)^T c_l     (the signals around the hypothesis: invariant under (R, c) -> (G R, D(G) c))
    v = MLP(flatten(c~), t) the body-frame velocity bivector (e12, e13, e23), last layer zero-initialised
    Z2 Reynolds average over the mirror F = diag(-1, 1, 1): v = 1/2 [m(c~) + S m(D(F) c~)], S = (-1, -1, +1).
Body velocity is invariant under R -> G R, so the Euler step r <- r exp(dt v) commutes with camera roll;
the flow of a mirrored image is the mirrored flow. Exact for 90 degree turns and the flip of the map
(tests/test_so3fourier.py); the ResNet in front is not equivariant.

canonicalize=False gives the non-equivariant twin for the fair test: same condition path, the read-out sees
[c, R] instead of D(R)^T c (and no mirror average), with the parameter count kept within ~0.3 %.

Conventions: e1 = image columns (right), e2 = rows (down), e3 = optical axis;
torch.rot90(map, 1, (2, 3)) <-> R -> R_z(-90 deg) R; left-right flip <-> R -> F R F.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def _o3():
    from e3nn import o3
    return o3


class _float64_default:
    """e3nn builds some constants in torch's default dtype; compute the fixed geometry in float64."""

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *exc):
        torch.set_default_dtype(self.prev)


def hemisphere_grid(nside: int = 4) -> torch.Tensor:
    """HEALPix pixel centres with z > 0, (P, 3). The grid is D4-symmetric about e3."""
    import healpy as hp
    x, y, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    pts = np.stack([x, y, z], -1)[z > 1e-9]
    return torch.tensor(pts, dtype=torch.float64)


def harmonic_basis(lmax: int, n_points: int = 64):
    """Fixed sphere points X (n, 3) and, per degree l, pinv(Y_l(X)) (2l+1, n), all float64.

    e3nn's real spherical harmonics rotate as Y_l(R x) = D^l(R) Y_l(x), so with the points stacked,
    D^l(R)^T = pinv(Y_l(X)) Y_l(X R^T): the Wigner matrix from a handful of harmonics of rotated points.
    """
    o3 = _o3()
    i = torch.arange(n_points, dtype=torch.float64) + 0.5          # Fibonacci sphere: well-conditioned for l <= 6
    z = 1 - 2 * i / n_points
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = (1 - z ** 2).sqrt()
    X = torch.stack([r * phi.cos(), r * phi.sin(), z], -1)
    with _float64_default():
        pinvs = [torch.linalg.pinv(o3.spherical_harmonics(l, X, normalize=True, normalization="component"))
                 for l in range(lmax + 1)]
    return X, pinvs


def wigner_from_harmonics(R: torch.Tensor, X: torch.Tensor, pinvs):
    """[D^l(R)] for rotations R (B, 3, 3), each (B, 2l+1, 2l+1), e3nn's convention (o3.wigner_D of
    o3.matrix_to_angles(R)). A polynomial in the entries of R: smooth everywhere, unlike the Euler-angle route,
    whose derivative blows up at gimbal lock (it made the flow log-likelihood NaN; reports/symsol_p1_stall.md)."""
    o3 = _o3()
    XR = torch.einsum("bij,nj->bni", R, X.to(R))                    # the points rotated by R
    out = []
    for l, P in enumerate(pinvs):
        Y = o3.spherical_harmonics(l, XR, normalize=True, normalization="component")    # (B, n, 2l+1)
        out.append(torch.einsum("kn,bnm->bmk", P.to(R), Y))          # D = (pinv(Y_X) Y_{XR^T})^T
    return out


def wigner_blocks(R: torch.Tensor, lmax: int, basis=None):
    """[D^l(R)]_{l <= lmax} for a batch of rotation matrices (B, 3, 3), each (B, 2l+1, 2l+1)."""
    X, pinvs = harmonic_basis(lmax) if basis is None else basis
    return wigner_from_harmonics(R, X, pinvs)


def n_coeffs(lmax: int) -> int:
    return sum((2 * l + 1) ** 2 for l in range(lmax + 1))


class SphereLift(nn.Module):
    """fmap (B, c_in, grid, grid) -> C_s scalar signals on the P points of a HEALPix hemisphere (B, C_s, P).

    The 1x1 conv and the D4-steerable orthographic lift of SphereConditionHead, without the Fourier part:
    shared with the rotor-correlation head (models/rotor_corr_head.py), which reads the points directly.
    """

    def __init__(self, c_in: int = 2048, c_s: int = 32, grid: int = 7, nside: int = 4, coverage: float = 0.9,
                 sigma: float = 0.2, taper_beta: float = math.radians(75), max_beta: float = math.radians(90)):
        super().__init__()
        if grid % 2 == 0 or grid < 3:
            raise ValueError("grid must be odd (>= 3) so the centre cell sits on the optical axis")
        self.c_s, self.grid = c_s, grid
        self.proj = nn.Conv2d(c_in, c_s, 1)
        pts = hemisphere_grid(nside)                                       # (P, 3)
        self.register_buffer("points", pts, persistent=False)
        max_r = pts[:, :2].norm(dim=1).max()
        self.register_buffer("q", pts[:, :2] / max_r, persistent=False)   # point -> image position in [-1, 1]
        off = (torch.arange(grid, dtype=torch.float64) - grid // 2) / (grid // 2)
        rows, cols = torch.meshgrid(off, off, indexing="ij")
        p = torch.stack([cols.flatten(), rows.flatten()], -1)             # (grid^2, 2): (u, v) = (col, row)
        self.register_buffer("p", p, persistent=False)
        rad2 = (p ** 2).sum(-1)
        uniq = rad2.unique()
        self.register_buffer("ridx", torch.searchsorted(uniq, rad2), persistent=False)
        self.radial_gain = nn.Parameter(torch.ones(len(uniq)))             # D4-invariant per-cell gain
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.coverage = nn.Parameter(torch.tensor(coverage))
        beta = torch.acos(pts[:, 2].clamp(-1, 1))
        taper = ((beta - max_beta) / (taper_beta - max_beta)).clamp(max=1.0)
        self.register_buffer("taper", taper, persistent=False)             # fades out towards the rim

    def lift_kernel(self, dtype):
        """(grid^2, P) weights from cell to hemisphere point."""
        p, q = self.p.to(dtype), self.q.to(dtype)
        d2 = ((p[:, None, :] - self.coverage.to(dtype) * q[None, :, :]) ** 2).sum(-1)
        k = torch.exp(-d2 / (2 * torch.exp(2 * self.log_sigma.to(dtype))))
        k = k / k.sum(0, keepdim=True)
        return k * self.radial_gain.to(dtype)[self.ridx][:, None] * self.taper.to(dtype)[None, :]

    def signal(self, fmap):
        b, _, h, w = fmap.shape
        if (h, w) != (self.grid, self.grid):
            raise ValueError(f"{type(self).__name__} expects a {self.grid}x{self.grid} map, got {h}x{w}")
        feats = self.proj(fmap).flatten(2)
        return torch.relu(feats @ self.lift_kernel(feats.dtype))           # (B, C_s, P)

    def forward(self, fmap):
        return self.signal(fmap)


class SphereConditionHead(SphereLift):
    """fmap (B, c_in, grid, grid) -> SO(3) Fourier coefficients (B, c_f, sum_l (2l+1)^2)."""

    def __init__(self, c_in: int = 2048, c_s: int = 32, c_f: int = 8, lmax: int = 6, grid: int = 7,
                 nside: int = 4, coverage: float = 0.9, sigma: float = 0.2,
                 taper_beta: float = math.radians(75), max_beta: float = math.radians(90)):
        super().__init__(c_in, c_s, grid, nside, coverage, sigma, taper_beta, max_beta)
        o3 = _o3()
        self.lmax, self.c_f = lmax, c_f
        pts = self.points
        with _float64_default():
            Y = o3.spherical_harmonics(list(range(lmax + 1)), pts, normalize=True, normalization="component")
        # Geometry buffers stay float64 (the D4 symmetry of the grid is exact only to their precision) and are
        # cast to the feature dtype at use.
        self.register_buffer("Y", Y.to(torch.float64), persistent=False)  # (P, (L+1)^2), real, natural parity
        self.l_gain = nn.Parameter(torch.ones(c_s, lmax + 1))              # isotropic learned S2 kernel
        self.psi = nn.ParameterList([
            nn.Parameter(torch.randn(c_s, c_f, 2 * l + 1) / math.sqrt(c_s * (2 * l + 1)))
            for l in range(lmax + 1)])                                     # S2 -> SO(3) filters, Fourier domain
        self.gate_a = nn.Parameter(torch.ones(c_f, lmax + 1))
        self.gate_b = nn.Parameter(torch.zeros(c_f, lmax + 1))

    @property
    def width(self) -> int:
        return n_coeffs(self.lmax)

    def s2_coeffs(self, fmap):
        sig = self.signal(fmap)                                             # (B, C_s, P)
        psi = sig @ self.Y.to(sig.dtype) / math.sqrt(self.Y.shape[0])    # equal-area grid: a plain mean
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]

    def forward(self, fmap):
        return self.so3_coeffs(self.s2_coeffs(fmap))

    def so3_coeffs(self, psi):
        """S2 coefficients (B, C_s, (L+1)^2) -> S2 -> SO(3) conv -> norm gate -> (B, C_f, sum_l (2l+1)^2)."""
        out, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            c = torch.einsum("bxm,xyn->bymn", psi[:, :, start:start + n], self.psi[l])   # (B, C_f, m, n)
            norm = c.flatten(2).norm(dim=-1) / n                                        # left/right invariant
            gate = torch.sigmoid(self.gate_a[:, l] * norm + self.gate_b[:, l])
            out.append((c * gate[..., None, None]).flatten(2))
            start += n
        return torch.cat(out, -1)


class SphereGATrHead(SphereConditionHead):
    """SphereConditionHead with a GATr encoder on the sphere (the GA-encoder + Fourier hybrid, for images).

    The lift is SO(3)-Fourier's (1x1 conv -> c_s channels -> Gaussian lift onto the P hemisphere points). The P points
    become GATr tokens: the point x_k as a PGA trivector and as a vector, its c_s lifted features as scalars. A few
    GATr blocks let every point see the others; each point's c_s output scalars g_c(k) weight the harmonics at its
    FIXED direction: psi_c,lm = sum_k g_c(k) Y_lm(x_k) / sqrt(P). Then the usual S2 -> SO(3) conv and gates.
    A 90 degree turn or a flip of the map permutes the points and acts on x_k by G, so GATr (equivariant, scalar
    outputs invariant) keeps the head exactly as equivariant as SphereConditionHead.
    """

    def __init__(self, c_in: int = 2048, c_s: int = 32, c_f: int = 8, lmax: int = 6, nside: int = 4,
                 mv_channels: int = 8, s_channels: int = 32, num_blocks: int = 2, num_heads: int = 4):
        super().__init__(c_in, c_s=c_s, c_f=c_f, lmax=lmax, nside=nside)
        from pose3d.models.gatr_denoiser import _VECTOR, _import_gatr
        GATr = _import_gatr()
        self.encoder = GATr(in_mv_channels=2, out_mv_channels=1, hidden_mv_channels=mv_channels, in_s_channels=c_s,
                            out_s_channels=c_s, hidden_s_channels=s_channels, attention={"num_heads": num_heads},
                            mlp={}, num_blocks=num_blocks)
        self.register_buffer("_vec_index", torch.tensor(_VECTOR), persistent=False)

    def s2_coeffs(self, fmap):
        from gatr.interface import embed_point
        sig = self.signal(fmap)                                             # (B, c_s, P)
        b, _, p = sig.shape
        pts = self.points.to(sig.dtype)
        mv = sig.new_zeros(b, p, 2, 16)
        mv[:, :, 0] = embed_point(pts).expand(b, -1, -1)
        mv[:, :, 1, self._vec_index] = pts
        _, g = self.encoder(mv, scalars=sig.transpose(1, 2))               # (B, P, c_s), invariant
        psi = g.transpose(1, 2) @ self.Y.to(sig.dtype) / math.sqrt(self.Y.shape[0])
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]


class PointSphereHead(SphereConditionHead):
    """A point cloud (B, N, 3) -> SO(3) Fourier coefficients, as SphereConditionHead does for an image.

    Each point contributes C_s radial features h(|x_i|) (a small MLP: how far out it sits) at its direction
    u_i = x_i / |x_i|, so the cloud becomes C_s signals on the whole sphere with Fourier coefficients
    psi_c,lm = mean_i h_c(|x_i|) Y_lm(u_i), l <= lmax. Turning the cloud by Q turns them by D^l(Q) exactly (no
    grid, no lift), and the S2 -> SO(3) convolution and read-out are SphereConditionHead's.
    """

    def __init__(self, c_s: int = 32, c_f: int = 8, lmax: int = 6, radial_hidden: int = 32):
        nn.Module.__init__(self)
        self.c_s, self.c_f, self.lmax = c_s, c_f, lmax
        self.radial = nn.Sequential(nn.Linear(1, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, c_s))
        self.l_gain = nn.Parameter(torch.ones(c_s, lmax + 1))
        self.psi = nn.ParameterList([
            nn.Parameter(torch.randn(c_s, c_f, 2 * l + 1) / math.sqrt(c_s * (2 * l + 1)))
            for l in range(lmax + 1)])
        self.gate_a = nn.Parameter(torch.ones(c_f, lmax + 1))
        self.gate_b = nn.Parameter(torch.zeros(c_f, lmax + 1))

    def s2_coeffs(self, points):
        o3 = _o3()
        r = points.norm(dim=-1, keepdim=True)                               # (B, N, 1)
        u = points / r.clamp(min=1e-6)
        Y = o3.spherical_harmonics(list(range(self.lmax + 1)), u, normalize=True, normalization="component")
        psi = torch.einsum("bnc,bnk->bck", self.radial(r), Y) / points.shape[1]
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]


class PointGATrSphereHead(PointSphereHead):
    """PointSphereHead with a GATr encoder instead of the per-point radial MLP (the GA-encoder + Fourier hybrid).

    The cloud's points are GATr tokens (the point as a PGA trivector, x_i as a vector, |x_i| as a scalar); a few
    GATr blocks let every point see the others, and each point's c_s output SCALARS (rotation invariant) weight
    the harmonics at its ORIGINAL direction: psi_c,lm = mean_i g_c(i) Y_lm(x_i / |x_i|). The directions stay the
    real geometry (harmonics of learned directions did not help: reports/frequency_hypothesis_ru.md); GATr only
    decides how much each point counts in each channel, from the whole shape. Then SphereConditionHead's S2 -> SO(3)
    conv; the read-out D(R)^T c is SO3FourierVelocity's. Exactly equivariant to any rotation of the cloud.
    """

    def __init__(self, c_s: int = 32, c_f: int = 8, lmax: int = 6, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 2, num_heads: int = 4):
        super().__init__(c_s=c_s, c_f=c_f, lmax=lmax)
        del self.radial
        from pose3d.models.gatr_denoiser import _VECTOR, _import_gatr
        GATr = _import_gatr()
        self.encoder = GATr(in_mv_channels=2, out_mv_channels=1, hidden_mv_channels=mv_channels, in_s_channels=1,
                            out_s_channels=c_s, hidden_s_channels=s_channels, attention={"num_heads": num_heads},
                            mlp={}, num_blocks=num_blocks)
        self.register_buffer("_vec_index", torch.tensor(_VECTOR), persistent=False)

    def s2_coeffs(self, points):
        from gatr.interface import embed_point
        o3 = _o3()
        b, n, _ = points.shape
        mv = points.new_zeros(b, n, 2, 16)
        mv[:, :, 0] = embed_point(points)
        mv[:, :, 1, self._vec_index] = points
        r = points.norm(dim=-1, keepdim=True)
        _, g = self.encoder(mv, scalars=r)                                  # (B, N, c_s), invariant
        u = points / r.clamp(min=1e-6)
        Y = o3.spherical_harmonics(list(range(self.lmax + 1)), u, normalize=True, normalization="component")
        psi = torch.einsum("bnc,bnk->bck", g, Y) / n
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]


class SO3FourierVelocity(nn.Module):
    """(R (B, 3, 3), t (B,), c (B, c_f, N)) -> body velocity bivector (B, 3), (e12, e13, e23)."""

    def __init__(self, c_f: int = 8, lmax: int = 6, hidden: int = 44, t_dim: int = 64,
                 flip_symmetrize: bool = True, canonicalize: bool = True):
        super().__init__()
        self.c_f, self.lmax, self.t_dim = c_f, lmax, t_dim
        self.canonicalize = bool(canonicalize)
        self.flip_symmetrize = bool(flip_symmetrize) and self.canonicalize
        n_in = c_f * n_coeffs(lmax) + (0 if self.canonicalize else 9)
        self.first = nn.Linear(n_in, hidden)        # a bank of `hidden` SO(3) correlations evaluated at R
        self.film = nn.Linear(t_dim, 2 * hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if self.flip_symmetrize:
            o3 = _o3()
            with _float64_default():
                abc = o3.matrix_to_angles(torch.diag(torch.tensor([1.0, -1.0, -1.0])))   # R_x(pi)
                DF = [((-1) ** l) * o3.wigner_D(l, *abc) for l in range(lmax + 1)]   # D^l(F), F = diag(-1, 1, 1)
            for l, D in enumerate(DF):
                self.register_buffer(f"DF{l}", D.to(torch.float64), persistent=False)
        self.register_buffer("S", torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64), persistent=False)
        if self.canonicalize:
            # Fixed points and harmonic pseudo-inverses for D(R) (wigner_from_harmonics); buffers, so they move
            # with the module (e3nn's wigner_D builds its constants on the CPU at every call).
            X, pinvs = harmonic_basis(lmax)
            self.register_buffer("basis_points", X, persistent=False)
            for l, P in enumerate(pinvs):
                self.register_buffer(f"basis_pinv{l}", P, persistent=False)

    def _t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([a.cos(), a.sin()], -1)

    def _blocks(self, c):
        out, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            out.append(c[:, :, start:start + n * n].view(c.shape[0], self.c_f, n, n))
            start += n * n
        return out

    def _mlp(self, x, t):
        g = self.first(x)
        gamma, beta = self.film(self._t_embed(t.to(x.dtype))).chunk(2, -1)
        return self.out(self.mlp(g * (1 + gamma) + beta))

    def harmonic_basis(self):
        return self.basis_points, [getattr(self, f"basis_pinv{l}") for l in range(self.lmax + 1)]

    def forward(self, R, t, c):
        if not self.canonicalize:
            return self._mlp(torch.cat([c.flatten(1), R.flatten(1)], 1), t)
        ct = [torch.einsum("bmk,bymn->bykn", D.to(c.dtype), cl)            # D^l(R)^T c_l
              for D, cl in zip(wigner_blocks(R, self.lmax, self.harmonic_basis()), self._blocks(c))]
        v = self._mlp(torch.cat([x.flatten(1) for x in ct], 1), t)
        if not self.flip_symmetrize:
            return v
        mirrored = torch.cat([torch.einsum("mk,bykn->bymn", getattr(self, f"DF{l}").to(x.dtype), x).flatten(1)
                              for l, x in enumerate(ct)], 1)
        return 0.5 * (v + self.S.to(v.dtype) * self._mlp(mirrored, t))
