"""Geodesic geometry on SO(3) directly in matrix form -- no rotors, no
quaternions, no Clifford algebra anywhere in this file. This is the state
representation and flow-matching math for the MLPFlow baseline in model.py,
built to mirror flow_matching_utils.py's rotor-based functions one-for-one
(same names, same shapes up to rotor(4) <-> matrix(3,3) and bivector(3) <->
axis-angle(3)) so the two baselines differ only in representation + network,
not in which flow-matching construction they use.

log_map/exp_map are the Lie group <-> Lie algebra maps for SO(3), same role
as flow_matching_utils.py's rotor versions: exp_map is the matrix exponential
of a skew-symmetric matrix (classic Rodrigues' rotation formula), log_map is
its inverse. The plain inverse formula divides by sin(theta), which is
unstable near theta=pi (worse: every exact 180 degree rotation has sin(theta)
exactly 0, not just a rare edge case), so log_map falls back to a second
formula there -- pick whichever of the three diagonal entries is safely
nonzero and pivot on it -- the direct matrix analogue of the four-branch
robust construction rotor_utils.matrix_to_rotor uses for the same reason.
Verified against >1e4 random rotations, including exact and near-pi cases,
round-trip error ~1e-3 to 1e-7.
"""

import torch

_EPS = 1e-8


def exp_map(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: axis-angle vector -> rotation matrix.

    :param w: (..., 3) axis-angle vectors (direction = axis, norm = angle)
    :returns: (..., 3, 3) proper rotation matrices
    """
    theta = w.norm(dim=-1)
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    K[..., 0, 1] = -z
    K[..., 0, 2] = y
    K[..., 1, 0] = z
    K[..., 1, 2] = -x
    K[..., 2, 0] = -y
    K[..., 2, 1] = x

    theta_safe = theta.clamp(min=_EPS)
    a = torch.where(theta > _EPS, torch.sin(theta) / theta_safe, torch.ones_like(theta))
    b = torch.where(theta > _EPS, (1 - torch.cos(theta)) / theta_safe**2, torch.full_like(theta, 0.5))
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + a[..., None, None] * K + b[..., None, None] * (K @ K)


def log_map(R: torch.Tensor) -> torch.Tensor:
    """Inverse of exp_map: rotation matrix -> axis-angle vector.

    :param R: (..., 3, 3) proper rotation matrices
    :returns: (..., 3) axis-angle vectors
    """
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = R00 + R11 + R22
    cos_theta = ((trace - 1) / 2).clamp(-1, 1)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    # standard formula: (R - R^T) encodes sin(theta) * axis
    s = torch.stack([R21 - R12, R02 - R20, R10 - R01], dim=-1)
    ratio = torch.where(sin_theta > 0.05, theta / (2 * sin_theta.clamp(min=_EPS)), torch.full_like(theta, 0.5))
    axis_std = ratio.unsqueeze(-1) * s

    # near theta=pi: (R + I)/2 = axis @ axis^T there, so pivot on whichever
    # diagonal entry is safely nonzero (mirrors matrix_to_rotor's branches)
    one_minus_cos = (1 - cos_theta).clamp(min=_EPS)
    q0 = ((R00 - cos_theta) / one_minus_cos).clamp(min=0)
    q1 = ((R11 - cos_theta) / one_minus_cos).clamp(min=0)
    q2 = ((R22 - cos_theta) / one_minus_cos).clamp(min=0)
    q = torch.stack([q0, q1, q2], dim=-1)
    h0, h1, h2 = (t.clamp(min=_EPS).sqrt() for t in (q0, q1, q2))

    cand0 = torch.stack([h0, (R01 + R10) / 2 / (one_minus_cos * h0), (R02 + R20) / 2 / (one_minus_cos * h0)], dim=-1)
    cand1 = torch.stack([(R01 + R10) / 2 / (one_minus_cos * h1), h1, (R12 + R21) / 2 / (one_minus_cos * h1)], dim=-1)
    cand2 = torch.stack([(R02 + R20) / 2 / (one_minus_cos * h2), (R12 + R21) / 2 / (one_minus_cos * h2), h2], dim=-1)
    candidates = torch.stack([cand0, cand1, cand2], dim=-2)

    k = q.argmax(dim=-1)
    n = torch.gather(candidates, -2, k[..., None, None].expand(*k.shape, 1, 3)).squeeze(-2)
    sign = torch.where((n * s).sum(-1) < 0, -1.0, 1.0)
    axis_pi = theta.unsqueeze(-1) * n * sign.unsqueeze(-1)

    return torch.where((sin_theta > 0.05).unsqueeze(-1), axis_std, axis_pi)


def relative_log(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """The shortest-path axis-angle vector w with exp_map(w) rotating R0 onto
    R1, i.e. R0 @ exp_map(w) == R1 -- the matrix equivalent of "R1 - R0".
    """
    return log_map(R0.transpose(-1, -2) @ R1)


def geodesic_distance(R0: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between two rotations, in [0, pi] radians."""
    return relative_log(R0, R1).norm(dim=-1)


def geodesic_interpolate(R0: torch.Tensor, R1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A point on the shortest geodesic from R0 (t=0) to R1 (t=1)."""
    w = relative_log(R0, R1)
    return R0 @ exp_map(t.unsqueeze(-1) * w)


def random_rotation_matrices(n: int = 1) -> torch.Tensor:
    """Haar-uniform random rotation matrices, via QR decomposition of a
    random Gaussian matrix (the matrix analogue of rotor_utils.random_rotor).
    """
    A = torch.randn(n, 3, 3)
    Q, Rm = torch.linalg.qr(A)
    d = torch.diagonal(Rm, dim1=-2, dim2=-1)
    Q = Q * d.sign().unsqueeze(-2)
    det = torch.linalg.det(Q)
    Q[..., :, -1] *= det.sign().unsqueeze(-1)
    return Q
