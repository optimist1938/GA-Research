"""Fused-rank forward for NativeSTFBlock: the same function and the same parameters, fewer GPU kernels.

MEASURED SLOWER (2026-10-30, valeriykryuchin/stf-fast-check-t4, T4, batch 16, FP32): 1.33 s/step vs 0.97 for the
reference block, peak 12.3 vs 7.9 GB, despite 6.7k vs 9.8k kernels. Zero-padding every rank to D = 13 and running the
feedback chain over all slots moves more bytes than the per-rank loop saves. Kept for reference, off by default.

The reference block (native_stf.py) loops over ranks 2..lmax in Python and gathers/scatters PGA blades with list
indices, so every elementwise op runs once per rank as its own memory-bound kernel (T4 profile: +5k kernels per step
for +6.6 % FLOPs, and a sort-based index_put backward). Here:
  * the ranks live in one zero-padded tensor T[b, p, C, L, D] (L = lmax - 1 ranks, D = 2 lmax + 1); padded entries
    stay exactly zero, so every reduction over D is unchanged;
  * per-rank linears are stacked on the fly from the block's own ModuleDicts (state_dict unchanged: checkpoints are
    interchangeable with the reference path);
  * the calibrated Cartesian up/down maps are stacked into [L, D, 3, D]; the feedback down-chains run as L masked
    stacked steps;
  * blade gathers/scatters are matmuls with fixed 0/1 matrices, the bivector reorder a fixed 3x3 matrix.
Enabled per field by NativeSTFCloudField.fast (env NATIVE_STF_FAST=1); NATIVE_STF_COMPILE=1 wraps the block in
torch.compile.
"""
import math
import os

import torch
import torch.nn.functional as F

from pose3d.models.gatr_denoiser import _ROTATION_BIVECTOR, _VECTOR


class StackedMaps:
    """The calibrated up/down maps of a CartesianMaps, stacked per rank slot and zero-padded to D."""

    def __init__(self, maps, lmax, device):
        L, D = lmax - 1, 2 * lmax + 1
        cal = lambda kind, l: (getattr(maps, f'{kind}{l}') * getattr(maps, f'{kind}_scale{l}')).detach().float().cpu()
        up = torch.zeros(L, D, 3, D)
        down = torch.zeros(L, D, 3, D)
        for i, l in enumerate(range(2, lmax + 1)):
            u = cal('up', l)                      # (d_{l-1}, 3, d_l): rank l-1 (slot i-1, or v for l = 2) -> rank l
            up[i, :u.shape[0], :, :u.shape[2]] = u
            if l < lmax:
                d = cal('down', l + 1)            # (d_{l+1}, 3, d_l): rank l+1 (slot i+1) -> rank l
                down[i, :d.shape[0], :, :d.shape[2]] = d
        chain, active = [], []
        for j in range(L):                        # step j takes slot i (rank l) from rank l - j to l - j - 1
            m = torch.zeros(L, D, 3, D)
            a = torch.zeros(L)
            for i, l in enumerate(range(2, lmax + 1)):
                r = l - j
                if r >= 2:
                    d = cal('down', r)
                    m[i, :d.shape[0], :, :d.shape[2]] = d
                    a[i] = 1
            chain.append(m)
            active.append(a)
        flat = lambda m: m.reshape(L, D, 3 * D).to(device)
        self.L, self.D = L, D
        self.up, self.down = flat(up), flat(down)
        self.chain = [flat(m) for m in chain]
        self.active = [a.view(1, 1, 1, L, 1).to(device) for a in active]
        self.dims = torch.tensor([2 * l + 1 for l in range(2, lmax + 1)], dtype=torch.float32, device=device)
        ev = torch.zeros(3, 16)
        eb = torch.zeros(3, 16)
        for i in range(3):
            ev[i, _VECTOR[i]] = 1
            eb[i, _ROTATION_BIVECTOR[i]] = 1
        perm = torch.zeros(3, 3)                  # bivector = (omega_z, -omega_y, omega_x)
        perm[2, 0], perm[1, 1], perm[0, 2] = 1, -1, 1
        self.gather_vector = ev.T.contiguous().to(device)   # (16, 3)
        self.embed_vector = ev.to(device)                    # (3, 16)
        self.embed_bivector = (perm @ eb).to(device)         # (3, 16), reorder folded in


def contract(X, M, vec):
    """out[b,p,c,l,o] = sum_{a,i} X[b,p,c,l,a] vec[b,p,c,i] M[l,a,i,o], M given as (L, D, 3 D)."""
    t = torch.einsum('bpcla,laj->bpclj', X, M)
    t = t.unflatten(-1, (3, X.shape[-1]))
    return (t * vec[:, :, :, None, :, None]).sum(-2)


def track_norm(T):
    """Reference track_norm per rank slot: one invariant radius per token and rank."""
    return T / (T.square().sum(-1).mean(2, keepdim=True)[..., None] + 1e-6).sqrt()


def pack(tracks, lmax):
    D = 2 * lmax + 1
    return torch.stack([F.pad(tracks[l], (0, D - tracks[l].shape[-1])) for l in range(2, lmax + 1)], dim=3)


def unpack(T, lmax):
    return {l: T[..., i, :2 * l + 1] for i, l in enumerate(range(2, lmax + 1))}


def _stack(md, lmax, attr='weight'):
    return torch.stack([getattr(md[str(l)], attr) for l in range(2, lmax + 1)])


def fast_block_forward(block, mv, scalars, T, fm):
    """NativeSTFBlock.forward(mv, scalars, tracks, maps) on the packed tracks T; returns (mv, scalars, T)."""
    lmax, C, L = block.lmax, block.channels, fm.L
    s = block.scalar_norm(scalars)
    Z = track_norm(T)
    w_qkv = torch.stack([_stack(block.q, lmax), _stack(block.k, lmax), _stack(block.val, lmax)], 1)   # (L, 3, C, C)
    qkv = torch.einsum('bpcld,lkc->bpkld', Z, w_qkv.reshape(L, 3 * C, C))
    q, k, value = qkv.split(C, dim=2)
    scale = (1.0 / torch.sqrt(fm.dims * C * (lmax - 1))).view(1, 1, 1, L, 1)
    score = block.qs(s) @ block.ks(s).transpose(-1, -2) / 4
    score = score + torch.einsum('bpcld,bncld->bpn', q * scale.to(q.dtype), k)
    attention = score.softmax(-1)
    vectors = mv @ fm.gather_vector.to(mv.dtype)
    v = block.v(vectors.transpose(-1, -2)).transpose(-1, -2)
    w = block.w(vectors.transpose(-1, -2)).transpose(-1, -2)
    v = v / (v.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    w = w / (w.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    message = torch.einsum('bpn,bncld->bpcld', attention, value)
    lower = torch.cat([F.pad(v, (0, fm.D - 3)).unsqueeze(3), Z[:, :, :, :L - 1]], dim=3)
    upward = contract(lower, fm.up, v)
    downward = contract(torch.roll(Z, -1, dims=3), fm.down, w)    # the wrapped last slot meets a zero map
    delta = torch.einsum('bpcld,lkc->bpkld', (message + upward + downward) / math.sqrt(3), _stack(block.mix, lmax))
    norms = (T.square().sum(-1) + 1e-8).sqrt()                     # (b, p, C, L)
    ln = torch.log1p(norms)
    wg, bg = _stack(block.gate, lmax), _stack(block.gate, lmax, 'bias')   # (L, C, S + C), (L, C)
    S = s.shape[-1]
    gate = torch.einsum('bps,lcs->bpcl', s, wg[:, :, :S]) + torch.einsum('bpkl,lck->bpcl', ln, wg[:, :, S:]) + bg.T
    gate = torch.sigmoid(gate)[..., None]
    T = T + block.track_step.view(1, 1, 1, L, 1) * gate * delta
    dot = (Z * message).sum(-1)
    invariants = torch.stack([ln, dot], -1).permute(0, 1, 3, 4, 2).reshape(*ln.shape[:2], 2 * L * C)
    scalars = scalars + block.s_step * block.scalar_out(invariants)
    fgate = (1 + torch.tanh(block.feedback_gate(block.scalar_norm(scalars))))[..., None, None]   # (b, p, C, 1, 1)
    X = track_norm(T)
    for m, a in zip(fm.chain, fm.active):
        X = a * contract(X, m, v) + (1 - a) * X
    vector = X[..., :3] * fgate                                     # (b, p, C, L, 3)
    polar = torch.einsum('bpcli,lmc->bpmi', vector, _stack(block.polar_out, lmax))
    omega = torch.linalg.cross(vector, w[:, :, :, None, :].expand_as(vector), dim=-1)
    axial = torch.einsum('bpcli,lmc->bpmi', omega, _stack(block.axial_out, lmax))
    delta_mv = (polar @ fm.embed_vector.to(polar.dtype) + axial @ fm.embed_bivector.to(axial.dtype)) / math.sqrt(L)
    return mv + block.mv_step * delta_mv, scalars, T


_COMPILED = None


def block_fn():
    """fast_block_forward, under torch.compile when NATIVE_STF_COMPILE=1."""
    global _COMPILED
    if os.environ.get('NATIVE_STF_COMPILE') == '1':
        if _COMPILED is None:
            import torch._dynamo
            # One graph per block object (4 per field, x train/eval, x dtype): keep them all cached.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            _COMPILED = torch.compile(fast_block_forward, dynamic=False)
        return _COMPILED
    return fast_block_forward
