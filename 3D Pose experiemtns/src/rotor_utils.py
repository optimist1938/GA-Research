import torch

_EPS = 1e-8

# Theorem 12 from https://arxiv.org/pdf/2412.02772
def matrix_to_rotor(R):
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    q = torch.stack([
        1 + R00 + R11 + R22,
        1 - R00 - R11 + R22,
        1 - R00 + R11 - R22,
        1 + R00 - R11 - R22,
    ], dim=-1).clamp(min=0)
    half = 0.5 * torch.sqrt(q.clamp(min=_EPS))
    h0, h1, h2, h3 = half[..., 0], half[..., 1], half[..., 2], half[..., 3]

    candidates = torch.stack([
        torch.stack([h0, (R01 - R10) / (4 * h0), (R02 - R20) / (4 * h0), (R12 - R21) / (4 * h0)], dim=-1),
        torch.stack([(R01 - R10) / (4 * h1), h1, -(R12 + R21) / (4 * h1), (R02 + R20) / (4 * h1)], dim=-1),
        torch.stack([(R02 - R20) / (4 * h2), -(R12 + R21) / (4 * h2), h2, -(R01 + R10) / (4 * h2)], dim=-1),
        torch.stack([(R12 - R21) / (4 * h3), (R02 + R20) / (4 * h3), -(R01 + R10) / (4 * h3), h3], dim=-1),
    ], dim=-2) 
    k = q.argmax(dim=-1)
    # picking the candidate for each sample
    rotor = torch.gather(candidates, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return rotor / rotor.norm(dim=-1, keepdim=True)


# rotor's sandwich action (algebra.sandwich / algebra.inverse) on the 3 basis vectors
def rotor_to_matrix(rotor, algebra):
    rotor = rotor / rotor.norm(dim=-1, keepdim=True)
    mv = embed_rotor(rotor, algebra)
    mv_inv = algebra.inverse(mv)

    cols = []
    for i in range(3):
        e_i = algebra.embed(rotor.new_ones(*rotor.shape[:-1], 1), (1 + i,))
        cols.append(algebra.get_grade(algebra.sandwich(mv, e_i, mv_inv), 1))
    return torch.stack(cols, dim=-1)


def random_rotor(n=1):
    raw = torch.randn(n, 4)
    return raw / raw.norm(dim=-1, keepdim=True)


ROTOR_BLADES = (0, 4, 5, 6)  # scalar, e12, e13, e23 -- a rotor's blades within algebra


def embed_rotor(rotor, algebra):
    return algebra.embed(rotor, ROTOR_BLADES)
