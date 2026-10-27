"""Rotor-correlation velocity field (GA-1): the SO(3)-Fourier head without Fourier transforms.

What makes the S2 -> SO(3) convolution of image2sphere work is template matching: a learned pattern psi on
the sphere is turned by the pose hypothesis R and compared with the image's signal f on the sphere,
[f * psi](R) = sum_i f(x_i) psi(R^T x_i). Here the same operation is written with rotors, in space:

    image (once per image): fmap -> 1x1 conv -> C_s scalar signals at the P hemisphere points x_i
                            (SphereLift, the lift of the SO(3)-Fourier head, models/so3fourier_head.py)
    template: J vectors p_j (body frame). Direction = where a bump sits, length kappa_j = |p_j| its sharpness.
    turn it by the pose: q_j = R p_j R~ (the rotor sandwich; computed as R @ p_j)
    compare: K_ji(R) = exp(<x_i, q_j> - |p_j|) = exp(kappa_j (cos angle(x_i, q_j) - 1))
             A_jc(R) = mean_i K_ji(R) f_c(x_i)
             score_o(R) = sum_jc W_ojc A_jc(R): `hidden` correlations [f * psi_o](R) with
                          psi_oc = sum_j W_ojc exp(<., p_j> - |p_j|), bumps of free position and width
    read-out: v = MLP(score, t) (FiLM on t, last layer zero-initialised), as SO3FourierVelocity.

Nothing here is band-limited: exp(kappa cos) has every angular frequency l (Bessel-weighted, significant up to
l ~ sqrt(2 kappa)), and the image never passes through a multivector, so the l <= 1 limit of a single Cl(3,0)
token does not apply. The score depends on (R, image) only through <x_i, R p_j>, so turning the image's points
and the pose together, x -> G x, R -> G R, changes nothing: the read-out is exactly invariant (any G, not
only the 90 degree turns of the map). Mirror: Z2 Reynolds average over F = diag(-1, 1, 1) as in
SO3FourierVelocity; on the flipped image F x_i at pose F R F the inner products are <x_i, R F p_j>, so the
mirrored term is the same model with the template reflected, v = 1/2 [m(p) + S m(F p)], S = (-1, -1, +1).
"""

import math

import torch
import torch.nn as nn


def fibonacci_sphere(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = (1 - z ** 2).sqrt()
    return torch.stack([r * phi.cos(), r * phi.sin(), z], -1)


class RotorCorrelationVelocity(nn.Module):
    """(R (B, 3, 3), t (B,), f (B, c_s, P)) -> body velocity bivector (B, 3), (e12, e13, e23)."""

    def __init__(self, points: torch.Tensor, c_s: int = 32, templates: int = 48, hidden: int = 103,
                 kappa: float = 10.0, t_dim: int = 64, flip_symmetrize: bool = True):
        super().__init__()
        self.c_s, self.t_dim = c_s, t_dim
        self.flip_symmetrize = bool(flip_symmetrize)
        self.register_buffer("points", points.to(torch.float64).clone(), persistent=False)   # (P, 3), cast at use
        self.p = nn.Parameter((kappa * fibonacci_sphere(templates)).float())                # (J, 3)
        self.W = nn.Parameter(torch.empty(hidden, templates, c_s))
        nn.init.uniform_(self.W, -1 / math.sqrt(templates * c_s), 1 / math.sqrt(templates * c_s))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.film = nn.Linear(t_dim, 2 * hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("F", torch.tensor([-1.0, 1.0, 1.0]), persistent=False)
        self.register_buffer("S", torch.tensor([-1.0, -1.0, 1.0]), persistent=False)

    def _t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([a.cos(), a.sin()], -1)

    def scores(self, R, f, p):
        """The `hidden` correlations of the image signal with the templates turned by R: (B, hidden)."""
        q = torch.einsum("bij,kj->bki", R, p)                                  # q_j = R p_j R~   (B, J, 3)
        K = torch.exp(q @ self.points.to(q).T - p.norm(dim=-1)[None, :, None])  # (B, J, P)
        A = torch.einsum("bjp,bcp->bjc", K, f) / f.shape[-1]
        return torch.einsum("bjc,ojc->bo", A, self.W) + self.bias

    def _mlp(self, g, t):
        gamma, beta = self.film(self._t_embed(t.to(g.dtype))).chunk(2, -1)
        return self.out(self.mlp(g * (1 + gamma) + beta))

    def forward(self, R, t, f):
        v = self._mlp(self.scores(R, f, self.p), t)
        if not self.flip_symmetrize:
            return v
        return 0.5 * (v + self.S * self._mlp(self.scores(R, f, self.p * self.F), t))
