"""GATr (Geometric Algebra Transformer) as the Clifford Flow denoiser and condition head.

`GATrVectorField` is a drop-in replacement for the CGENN-style vector field of `CliffordFlow`:
same (B, 2 + n_cond_mv, 8) Cl(3,0) input, same (B, 8) output. The multivectors are embedded into
the projective algebra Cl(3,0,1) that GATr works in, the rotor / time / condition multivectors
become tokens of one sequence, and the velocity is read from the rotor token's rotation bivector.

`GATrConditionHead` replaces the condition head the same way: (B, n_in, 8) backbone multivectors
in, (B, n_out, 8) condition multivectors out.

https://github.com/Qualcomm-AI-research/geometric-algebra-transformer

The reference package hard-imports `xformers` (only used for optional attention masks, which
this model never passes). A stub is installed when it is missing, so torch is never replaced.
"""

import sys
import types

import torch
import torch.nn as nn

# Cl(3,0) blade -> PGA Cl(3,0,1) blade, both in grade-major order:
#   1, e1, e2, e3, e12, e13, e23, e123  ->  1, e1..e3 (2-4), e12..e23 (8-10), e123 (14)
_PGA_INDEX = (0, 2, 3, 4, 8, 9, 10, 14)
_ROTATION_BIVECTOR = (8, 9, 10)  # e12, e13, e23 within the 16 PGA blades
_VECTOR = (2, 3, 4)              # e1, e2, e3 within the 16 PGA blades


def _ensure_xformers_stub():
    try:
        import xformers.ops  # noqa: F401
        return
    except ImportError:
        pass

    class AttentionBias:  # never instantiated: no attention mask is ever passed
        pass

    def memory_efficient_attention(*args, **kwargs):
        raise RuntimeError("xformers is not installed; GATr attention masks are unsupported")

    ops = types.ModuleType("xformers.ops")
    ops.AttentionBias = AttentionBias
    ops.memory_efficient_attention = memory_efficient_attention
    pkg = types.ModuleType("xformers")
    pkg.ops = ops
    sys.modules["xformers"], sys.modules["xformers.ops"] = pkg, ops


def _import_gatr():
    _ensure_xformers_stub()
    try:
        from gatr import GATr
    except ImportError as e:
        raise ImportError(
            "--vector_field gatr needs the GATr package: pip install --no-deps einops opt_einsum "
            "git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git"
        ) from e
    return GATr


class GATrVectorField(nn.Module):
    """(B, 2 + n_cond_mv, 8) Cl(3,0) multivectors -> (B, 1, 8), like TralaleroTralala.

    Token 0 carries the noisy rotor R_t, tokens 1.. the conditioning multivectors (token 1 is
    the time embedding, which the flow builds as a scalar multivector). Scalar channels tell the
    tokens apart: the time t, broadcast to every token, and a query flag that is 1 on token 0.
    With n_pose_tokens=3 the pose is the frame R e1, R e2, R e3 on tokens 0-2 (time on token 3),
    each with its own flag so the network knows which axis is which. With pose_channels=3 the frame
    is instead three multivector channels of token 0 (input (B, n, 3, 8)), so per-token geometric
    products can combine R e_i with image directions attention brings in, from the first block.
    """

    def __init__(self, n_cond_mv: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, n_pose_tokens: int = 1,
                 pose_channels: int = 1):
        super().__init__()
        # 1: the rotor token; 3: the frame R e1, R e2, R e3 (CliffordFlow pose_tokens='frame').
        # The time token follows the pose tokens; the query flag and the read-out stay on token 0.
        self.n_pose_tokens = n_pose_tokens
        self.pose_channels = pose_channels
        GATr = _import_gatr()
        self.net = GATr(
            in_mv_channels=pose_channels, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=1 + n_pose_tokens, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)
        self.register_buffer("_rot_index", torch.tensor(_ROTATION_BIVECTOR), persistent=False)
        self.register_buffer("_vec_index", torch.tensor(_VECTOR), persistent=False)
        self.out = self.net.linear_out  # zeroed by CliffordFlow so the initial velocity is 0

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(2)                     # (B, n, channels, 8)
        b, n, c, _ = x.shape
        pga = x.new_zeros(b, n, c, 16)
        pga[..., self._pga_index] = x
        scalars = x.new_zeros(b, n, 1 + self.n_pose_tokens)
        scalars[..., 0] = x[:, self.n_pose_tokens, 0, 0].unsqueeze(1)  # t: scalar part of the time token
        for i in range(self.n_pose_tokens):
            scalars[:, i, 1 + i] = 1.0
        out_mv, _ = self.net(pga, scalars=scalars)  # (B, n, 1, 16)
        query = out_mv[:, 0, 0]  # (B, 16)
        result = x.new_zeros(b, 1, x.shape[-1])
        if self.n_pose_tokens == 1 and self.pose_channels == 1:
            result[:, 0, 4:7] = query[:, self._rot_index]  # e12, e13, e23 of Cl(3,0)
        else:
            # Frame tokens carry no bivector part, so at the zero-initialised read-out a bivector
            # read-out sees almost no signal and training stalls at a saddle. Read the vector part
            # instead (token 0 is R e1 itself) and take its dual e123 w, still a bivector under
            # rotations: e12 = w3, e13 = -w2, e23 = w1.
            w = query[:, self._vec_index]
            result[:, 0, 4:7] = torch.stack([w[:, 2], -w[:, 1], w[:, 0]], dim=-1)
        return result


class GATrConditionHead(nn.Module):
    """(B, n_in, 8) Cl(3,0) multivectors -> (B, n_out, 8), like TralaleroTralala.

    The n_in inputs are tokens, followed by n_out learned query multivectors; the output is read
    from the query positions, which is how GATr (no cross-attention) maps n_in tokens to n_out.
    The inputs are chunks of the pooled backbone vector, so which chunk is which matters; GATr
    treats tokens as an unordered set, hence every token also gets a learned scalar embedding.
    """

    def __init__(self, n_in: int, n_out: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, token_dim: int = 16):
        super().__init__()
        GATr = _import_gatr()
        self.net = GATr(
            in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=token_dim, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.n_in, self.n_out = n_in, n_out
        self.query = nn.Parameter(0.1 * torch.randn(n_out, 8))
        self.token_embedding = nn.Parameter(torch.randn(n_in + n_out, token_dim))
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)

    def forward(self, x):
        b = x.shape[0]
        tokens = torch.cat([x, self.query.to(x.dtype).expand(b, -1, -1)], dim=1)
        pga = x.new_zeros(b, self.n_in + self.n_out, 16)
        pga[..., self._pga_index] = tokens
        scalars = self.token_embedding.to(x.dtype).expand(b, -1, -1)
        out_mv, _ = self.net(pga.unsqueeze(2), scalars=scalars)  # (B, n_in + n_out, 1, 16)
        return out_mv[:, self.n_in:, 0][..., self._pga_index]
