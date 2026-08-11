import torch

from src.rotor_utils import ROTOR_BLADES

_EPS = 1e-8

# these are the two functions to make David Ruhe's framework make computations only in even-grades, where rotors live

def rotor_multiply(a, b, algebra):
    idx = torch.tensor(ROTOR_BLADES, device=a.device)
    return algebra.geometric_product(a, b, blades=(idx, idx, idx))


def rotor_conjugate(rotor, algebra):
    idx = torch.tensor(ROTOR_BLADES, device=rotor.device)
    return algebra.beta(rotor, blades=idx)


def log_map(rotor):
    rotor = torch.where(rotor[..., :1] < 0, -rotor, rotor)
    w, v = rotor[..., 0], rotor[..., 1:]
    vnorm = v.norm(dim=-1)
    theta = 2 * torch.atan2(vnorm, w)
    ratio = torch.where(vnorm > _EPS, theta / vnorm.clamp(min=_EPS), 2 / w.clamp(min=_EPS))
    return ratio.unsqueeze(-1) * v


def exp_map(bivector):
    theta = bivector.norm(dim=-1)
    half = 0.5 * theta
    scale = torch.where(theta > _EPS, torch.sin(half) / theta.clamp(min=_EPS), torch.full_like(theta, 0.5))
    return torch.cat([torch.cos(half).unsqueeze(-1), scale.unsqueeze(-1) * bivector], dim=-1)


def relative_log(a, b, algebra):
    return log_map(rotor_multiply(rotor_conjugate(a, algebra), b, algebra))


def geodesic_distance(a, b, algebra):
    return relative_log(a, b, algebra).norm(dim=-1)


def geodesic_interpolate(a, b, t, algebra):
    B = relative_log(a, b, algebra)
    return rotor_multiply(a, exp_map(t.unsqueeze(-1) * B), algebra)


def geodesic_mse_loss(pred, target, algebra):
    return relative_log(pred, target, algebra).pow(2).sum(-1).mean()
