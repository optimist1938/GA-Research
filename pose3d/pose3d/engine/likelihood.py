"""Exact log-likelihood of a flow-matching pose model, log p(R | image), for SYMSOL.

The flows start from the uniform distribution on SO(3) and move a pose by r <- r exp(dt v(r, t)), v the
body-frame velocity. Along a trajectory the density obeys d/dt log p_t(r_t) = -div v(r_t, t) (instantaneous
change of variables, Chen et al. 2018), so

    log p_1(R) = log p_0 - int_0^1 div v(r_t, t) dt,    r_1 = R,

integrated backwards from R (midpoint rule). The body-frame components v^i multiply left-invariant fields,
which are divergence-free for the Haar measure, so div v = sum_i d/de_i v^i(r exp(e)) at e = 0: three
backward passes through the small vector field per step, never through the backbone. The divergence is
independent of how the Lie algebra is scaled, so it needs no convention beyond the model's own exp.

Units follow IPDF / Image2Sphere / Howell et al.: SO(3) has volume pi^2, so the uniform density is
log(1 / pi^2) = -2.29 and values above it mean the model puts more density on the true pose than a guess.
"""

import math

import torch

import pose3d.geometry.matrix as matrix_flow
from pose3d.geometry.flow import exp_map as rotor_exp, rotor_multiply
from pose3d.geometry.rotor import matrix_to_rotor, rotor_to_matrix

LOG_UNIFORM = -2.0 * math.log(math.pi)


def _hat(w):
    K = w.new_zeros(*w.shape[:-1], 3, 3)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    return K


class _MatrixStates:
    """DiTFlow / MLPFlow: the pose is a rotation matrix, R <- R exp(dt w)."""

    def __init__(self, model):
        pass

    def from_matrix(self, R):
        return R

    def to_matrix(self, R):
        return R

    def step(self, R, w):
        return R @ matrix_flow.exp_map(w)

    def tangent(self, R, w):
        # Same first derivative at w = 0 as exp, without the NaN gradient of |w| at zero.
        return R @ (torch.eye(3, dtype=R.dtype, device=R.device) + _hat(w))


class _RotorStates:
    """CliffordFlow: the pose is a Spin(3) rotor (scalar, e12, e13, e23), r <- r exp(dt b)."""

    def __init__(self, model):
        self.algebra = model.algebra

    def from_matrix(self, R):
        return matrix_to_rotor(R)

    def to_matrix(self, r):
        return rotor_to_matrix(r, self.algebra)

    def step(self, r, b):
        return rotor_multiply(r, rotor_exp(b), self.algebra)

    def tangent(self, r, b):
        # exp(b) = cos|b|/2 + sin(|b|/2) b/|b| = 1 + b/2 + O(|b|^2).
        return rotor_multiply(r, torch.cat([torch.ones_like(b[..., :1]), 0.5 * b], -1), self.algebra)


def flow_states(model):
    return _RotorStates(model) if type(model).__name__ == "CliffordFlow" else _MatrixStates(model)


def _velocity_and_divergence(model, states, s, t, cond):
    with torch.enable_grad():
        e = torch.zeros(s.shape[0], 3, dtype=torch.float32, device=s.device, requires_grad=True)
        v = model.velocity(states.tangent(s.detach(), e), t, cond)
        div = sum(torch.autograd.grad(v[:, i].sum(), e, retain_graph=i < 2)[0][:, i] for i in range(3))
    return v.detach(), div.detach()


@torch.no_grad()
def flow_log_likelihood(model, cond, R, steps: int = 50, states=None):
    """log p(R | condition) for poses R (n, 3, 3), cond already repeated to n rows.

    The divergence takes three autograd passes with retain_graph through the velocity. A torch.compiled
    vector field (FlowConfig.gatr_compile) refuses that: its backward "was compiled with non-empty donated
    buffers", which killed the first SYMSOL RTX runs at the final evaluation. So the whole integration runs
    with torch.compile forced to eager (`torch.compiler.set_stance("force_eager")`): same function, same
    weights, no compiled graph.
    """
    with torch.compiler.set_stance("force_eager"):
        return _flow_log_likelihood(model, cond, R, steps=steps, states=states)


def _flow_log_likelihood(model, cond, R, steps, states):
    states = flow_states(model) if states is None else states
    s = states.from_matrix(R)
    div_integral = torch.zeros(R.shape[0], device=R.device)
    dt = 1.0 / steps
    for i in range(steps, 0, -1):
        t = torch.full((R.shape[0],), i * dt, device=R.device)
        k1 = model.velocity(s, t, cond)
        mid = states.step(s, -0.5 * dt * k1)
        v_mid, div = _velocity_and_divergence(model, states, mid, t - 0.5 * dt, cond)
        div_integral += dt * div
        s = states.step(s, -dt * v_mid)
    return LOG_UNIFORM - div_integral


@torch.no_grad()
def symsol_log_likelihood(model, loader, device, n_equivalents: int = 8, steps: int = 50, seed: int = 0):
    """IPDF's SYMSOL metric: the mean over images of the mean log p over each image's equivalent poses.

    A random subset of `n_equivalents` of each image's equivalents is scored (all of them when it has fewer):
    an unbiased estimate of the mean over the full set (360 poses for the cone, 720 for the cylinder).
    Returns (per-image log-likelihoods, per-image class indices).
    """
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    lls, classes = [], []
    for batch in loader:
        img = batch["img"].to(device)
        rots, n_eq = batch["rots_eq"], batch["n_eq"]
        b = img.shape[0]
        idx = torch.stack([torch.randperm(int(n), generator=gen)[:n_equivalents].repeat(
            math.ceil(n_equivalents / min(int(n), n_equivalents)))[:n_equivalents] for n in n_eq])
        picked = rots[torch.arange(b)[:, None], idx].reshape(-1, 3, 3).to(device)        # (b * k, 3, 3)
        weight = torch.stack([(torch.arange(n_equivalents) < min(int(n), n_equivalents)).float()
                              for n in n_eq])                                          # repeats count once
        cond = model.condition(img)
        cond = cond.repeat_interleave(n_equivalents, dim=0)
        ll = flow_log_likelihood(model, cond, picked, steps=steps).view(b, n_equivalents).cpu()
        lls.append((ll * weight).sum(1) / weight.sum(1))
        classes.append(batch["cls_eval"].view(-1))
    return torch.cat(lls).numpy(), torch.cat(classes).numpy()
