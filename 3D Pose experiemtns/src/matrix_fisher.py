import torch
import numpy as np

# Ported from Liu et al. 2023 (github.com/PKU-EPIC/RotationNormFlow, utils/fisher.py),
# dropping the pytorch3d/nflows/scipy dependencies we don't need.


def quat_to_rotmat(quat):
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (wy + xz),
        2 * (wz + xy), w2 - x2 + y2 - z2, 2 * (yz - wx),
        2 * (xz - wy), 2 * (wx + yz), w2 - x2 - y2 + z2,
    ], dim=-1).view(*quat.shape[:-1], 3, 3)
    return R


def proper_svd(A):
    U, S, Vh = torch.linalg.svd(A)
    detU, detV = torch.linalg.det(U), torch.linalg.det(Vh)
    U, S, Vh = U.clone(), S.clone(), Vh.clone()
    U[..., :, 2] *= detU.unsqueeze(-1)
    S[..., 2] *= detU * detV
    Vh[..., 2, :] *= detV.unsqueeze(-1)
    return U, S, Vh


def norm_approx(S):
    return 1.0 / torch.sqrt(8 * torch.pi * (S[..., 0] + S[..., 1]) * (S[..., 1] + S[..., 2]) * (S[..., 0] + S[..., 2]))


def log_prob(A, R):
    _, S, _ = proper_svd(A)
    trace = (R * A).sum(-1).sum(-1)
    return trace - S.sum(-1) - norm_approx(S).log()


def _sample_bingham(A4, Omega, std, M_star, n, oversample=8):
    device = A4.device
    while True:
        eps = torch.randn(n * oversample, 4, device=device)
        y = std * eps
        s = y / y.norm(dim=-1, keepdim=True)
        p_bing = torch.exp(-(s * A4 * s).sum(-1))
        p_acg = ((s * Omega * s).sum(-1)) ** (-2)
        accept = torch.rand(n * oversample, device=device) < p_bing / (M_star * p_acg)
        if accept.sum() >= n:
            return s[accept][:n]


def sample_one(A, b=1.5, oversample=8):
    # A: (3, 3) -> single (3, 3) rotation. Rejection sampling isn't
    # differentiable, so this always runs on a detached A.
    U, S, Vh = proper_svd(A.detach())

    A4 = torch.zeros(4, device=A.device, dtype=A.dtype)
    A4[1] = 2 * (S[1] + S[2])
    A4[2] = 2 * (S[0] + S[2])
    A4[3] = 2 * (S[0] + S[1])

    Omega = 1 + 2 * A4 / b
    std = Omega ** -0.5
    M_star = np.exp(-(4 - b) / 2) * (4 / b) ** 2

    quat = _sample_bingham(A4, Omega, std, M_star, n=1, oversample=oversample)
    R = quat_to_rotmat(quat)[0]
    return U @ R @ Vh


@torch.no_grad()
def sample_batch(A):
    # A: (N, 3, 3) -> (N, 3, 3), one sample per row. Used only as r0, an
    # initial guess drawn from a frozen, pretrained Fisher head.
    return torch.stack([sample_one(a) for a in A], dim=0)
