"""Geometric-algebra behavioural model for a concurrent multi-band PA.

Structure
---------
The reference memory polynomial is

.. math:: y(k) = \\mathrm{BL} * \\sum_m g_m(k)\\, x_1(k - s_m),
          \\qquad g_m(k) = \\sum_{\\mathbf n} c_{m\\mathbf n} \\prod_p T_{n_p}(|x_p(k - \\tau_{pm})|),

i.e. a complex gain ``g_m`` built from a *separable polynomial in the envelope
magnitudes only*, multiplying a delayed copy of the carrier.

:class:`CliffordPAModel` keeps the outer form and replaces ``g_m`` with a
Clifford-group-equivariant network over the delayed envelopes as
``Cl(2, 0)`` multivectors.  Two facts make this exact rather than decorative:

* A baseband phase rotation ``z -> z e^{j\\varphi}`` is the rotor sandwich
  ``v -> R v R~`` on the vector ``v = Re(z) e_1 + Im(z) e_2``.
* In ``Cl(2, 0)`` the even subalgebra is commutative and commutes with every
  rotor, so the even part of a multivector is *invariant* under that rotation --
  and it is isomorphic to ``C``.

So the even part of the network's output is precisely "a complex gain that does
not change when you rotate the constellation": the AM/AM + AM/PM characteristic,
obtained as a symmetry property instead of by restricting the basis to ``|x|``.
The model is then strictly more expressive than the reference, because ``g_m``
may depend on the relative *phases* between bands and taps, not just magnitudes,
while still being exactly phase-equivariant overall.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .ga import CliffordAlgebra, MVBlock, MVLinear
from .models import band_limit

__all__ = ["CliffordPAModel"]


class CliffordPAModel(nn.Module):
    """Clifford-equivariant PA behavioural model.

    Parameters
    ----------
    n_bands
        Number of concurrent carriers ``D``.  Band 0 is the one being modelled.
    env_taps
        Delays of the envelope samples fed to the network, e.g. ``range(-2, 9)``.
    linear_taps
        Delays ``s_m`` of the carrier copies the gains multiply.
    channels, n_blocks
        Width and depth of the geometric-product trunk.
    bl
        Band-limitation FIR taps, as in the reference model.
    scale
        Input scaling applied to the embedded envelopes.  The envelopes are
        unit-peak, and geometric-product blocks are quadratic, so a mild scale-up
        keeps the first block out of the flat region of its gates.
    """

    def __init__(
        self,
        n_bands: int,
        env_taps=range(-2, 9),
        linear_taps=(0, 1, 2),
        channels: int = 16,
        n_blocks: int = 2,
        bl: np.ndarray | None = None,
        scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.algebra = CliffordAlgebra(2, 0)
        self.n_bands = n_bands
        self.env_taps = list(env_taps)
        self.linear_taps = list(linear_taps)
        self.scale = scale

        n_in = n_bands * len(self.env_taps)
        self.embed = MVLinear(self.algebra, n_in, channels)
        self.blocks = nn.ModuleList(MVBlock(self.algebra, channels) for _ in range(n_blocks))
        self.readout = MVLinear(self.algebra, channels, len(self.linear_taps))
        # Start near the identity-free solution: zero readout means y = 0, so the
        # first gradient steps grow the model out of the no-model NMSE floor
        # instead of having to first undo a large random output.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

        bl_t = torch.ones(1) if bl is None else torch.as_tensor(np.asarray(bl), dtype=torch.float64)
        self.register_buffer("bl", bl_t)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def gains(self, x: torch.Tensor) -> torch.Tensor:
        """Invariant complex gains ``g_m(k)``; ``x`` is ``(D, N)`` complex.

        Returns ``(len(linear_taps), N)`` complex.
        """
        if x.ndim != 2 or x.shape[0] != self.n_bands:
            raise ValueError(f"expected x of shape ({self.n_bands}, N), got {tuple(x.shape)}")
        real_dtype = torch.float64 if x.dtype == torch.complex128 else torch.float32
        mv = torch.stack(
            [
                self.algebra.embed_complex(torch.roll(x[p], t, dims=0) * self.scale)
                for p in range(self.n_bands)
                for t in self.env_taps
            ],
            dim=-2,
        ).to(real_dtype)  # (N, C_in, 4)

        h = self.embed(mv)
        for block in self.blocks:
            h = block(h)
        out = self.readout(h)  # (N, len(linear_taps), 4)
        # Even part (blades 0 and e12) -- invariant under the rotor action.
        return torch.complex(out[..., 0], out[..., 3]).transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(D, N)`` complex, unit-peak rows.  Returns ``(N,)`` complex."""
        g = self.gains(x).to(x.dtype)
        y = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        for i, s in enumerate(self.linear_taps):
            y = y + g[i] * torch.roll(x[0], s, dims=0)
        return band_limit(y, self.bl) if self.bl.numel() > 1 else y
