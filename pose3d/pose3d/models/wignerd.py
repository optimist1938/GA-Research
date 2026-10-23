"""Direct Wigner-D harmonics prediction (Lee & Cho, "3D Equivariant Pose Regression via Direct Wigner-D Harmonics Prediction", NeurIPS 2024, arXiv 2411.00543).

The network is the Image2Sphere trunk (ResNet -> 1x1 conv -> orthographic projection onto an S2 HEALPix grid -> S2 convolution ->
SO(3) group convolution), whose output is a vector of M = sum_l (2l+1)^2 = 455 Wigner-D coefficients (lmax = 6). Image2Sphere reads that
vector as a function on SO(3) and trains it with a cross-entropy over a rotation grid. Here it is regressed directly:

    loss = sum_l w_l * || Psi_l - D^l(R_gt) ||^2          (the paper's frequency-domain MSE, eq. 4)

so no grid is touched in training. Inference is unchanged: the prediction is the grid rotation R maximising <Psi, D(R)> over an SO(3) HEALPix
grid (level 5, 2.36M rotations, 1.875 deg in the paper); the argmax of the softmax of those scores. The paper's optional gradient ascent is not implemented.

The target is built with the same `flat_wigner` the grid uses, so training and inference agree on the Euler convention whatever it is.
"""

import torch

from pose3d.models.i2s_real import I2SReal


def degree_weights(lmax: int, mode: str) -> torch.Tensor:
    """w_l repeated over the (2l+1)^2 coefficients of degree l, (M,)."""
    if mode not in ("unit", "uniform"):
        raise ValueError(f"loss_weights must be 'unit' or 'uniform', got {mode!r}")
    return torch.cat([torch.full(((2 * l + 1) ** 2,), 1.0 / (2 * l + 1) ** 2 if mode == "unit" else 1.0) for l in range(lmax + 1)])


class WignerD(I2SReal):
    def __init__(self, *args, lmax: int = 6, loss_weights: str = "unit", loss_reduction: str = "sum", eval_rec_level: int = 5, **kwargs):
        super().__init__(*args, lmax=lmax, eval_rec_level=1, **kwargs)       # the level-5 grid is built lazily, on the device that scores
        self.lmax = lmax
        self.eval_level = eval_rec_level
        self._grid = None
        if loss_reduction not in ("sum", "mean"):
            raise ValueError(f"loss_reduction must be 'sum' or 'mean', got {loss_reduction!r}")
        w = degree_weights(lmax, loss_weights)
        self.register_buffer("loss_w", w / w.numel() if loss_reduction == "mean" else w, persistent=False)

    @torch.no_grad()
    def _eval_grid(self, device):
        """(Wigner matrix (M, Q), rotations (Q, 3, 3)) of the SO(3) HEALPix grid, built in chunks on `device` and cached.

        Level 5 is 2.36M rotations, a 4.3 GB float32 matrix; building it in one go on the CPU needs several times that.
        """
        if self._grid is None or self._grid[0].device != torch.device(device):
            from e3nn import o3
            from image2sphere import so3_utils

            a, b, g = (t.to(device) for t in so3_utils.so3_healpix_grid(rec_level=self.eval_level))
            q = a.numel()
            wig = torch.empty(self.loss_w.numel(), q, device=device)
            for i in range(0, q, 65536):
                wig[:, i:i + 65536] = so3_utils.flat_wigner(self.lmax, a[i:i + 65536], b[i:i + 65536], g[i:i + 65536]).T
            self._grid = (wig, o3.angles_to_matrix(a, b, g))
        return self._grid

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """argmax over the grid of <Psi, D(R)> (the softmax of the paper's inference changes nothing for the argmax)."""
        wig, rots = self._eval_grid(img.device)
        psi = self.forward(img).squeeze(1).float()
        return rots[(psi @ wig).argmax(dim=1)]

    @torch.no_grad()
    def grid_logits(self, img: torch.Tensor) -> torch.Tensor:
        """Scores <Psi, D(R)> of every grid rotation, (B, Q): the unnormalised log-density the softmax of the paper's inference is taken over."""
        wig, _ = self._eval_grid(img.device)
        return self.forward(img).squeeze(1).float() @ wig

    def target(self, rot_gt: torch.Tensor) -> torch.Tensor:
        """Flattened Wigner-D coefficients of the ground-truth rotations, (B, M)."""
        from e3nn import o3
        from image2sphere import so3_utils

        # The augmented training rotations are not exactly orthonormal (e3nn asserts det == 1 to 1e-8): take the nearest rotation first.
        u, _, vh = torch.linalg.svd(rot_gt.detach().double().cpu())
        sign = torch.det(u @ vh)
        u = torch.cat([u[..., :2], u[..., 2:] * sign[:, None, None]], dim=-1)
        abg = o3.matrix_to_angles((u @ vh))              # CPU, as upstream: e3nn's angle extraction is faster and exact there
        return so3_utils.flat_wigner(self.lmax, *abg).float().to(rot_gt.device)

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion=None) -> torch.Tensor:
        # criterion is ignored, like for the grid cross-entropy it replaces.
        psi = self.forward(img).squeeze(1).float()
        return (self.loss_w * (psi - self.target(rot_gt)) ** 2).sum(-1).mean()
