"""MLPFlow: the non-equivariant flow-matching baseline (plain MLPs on rotation matrices)."""

import torch
import torch.nn as nn

import pose3d.geometry.matrix as matrix_flow
from pose3d.models.clifford_flow import ImageToMultivectors


class MLPFlow(nn.Module):
    def __init__(self, algebra, cond_hidden: int = 210, cond_features: int = 512, field_hidden: int = 128,
                 pretrained_backbone: bool = False, adapter_grid: int = 16, encoder_type: str = "resnet",
                 conv_adapter: bool = True):
        super().__init__()
        # conv_adapter=False: the pooled backbone vector cut into multivectors, the input DiTFlow sees.
        self.adapter = ImageToMultivectors(algebra, grid=adapter_grid,
                                           pretrained_backbone=pretrained_backbone,
                                           encoder_type=encoder_type, conv_adapter=conv_adapter)
        adapter_dim = self.adapter.n_mv * (2**algebra.dim)

        self.condition_mlp = nn.Sequential(
            nn.Linear(adapter_dim, cond_hidden), nn.ReLU(),
            nn.Linear(cond_hidden, cond_features),
        )
        self.field_mlp = nn.Sequential(
            nn.Linear(9 + 1 + cond_features, field_hidden), nn.ReLU(),
            nn.Linear(field_hidden, 3),
        )
        # zero-init so the flow starts as the identity map, same as CliffordFlow
        nn.init.zeros_(self.field_mlp[-1].weight)
        nn.init.zeros_(self.field_mlp[-1].bias)

    def condition(self, x):
        mv = self.adapter(x)
        return self.condition_mlp(mv.flatten(1))

    def velocity(self, R, t, cond):
        inp = torch.cat([R.flatten(1), t.view(-1, 1), cond], dim=1)
        return self.field_mlp(inp)

    def forward(self, x, R, t):
        return self.velocity(R, t, self.condition(x))

    def compute_loss(self, img, rot_gt, criterion=None):
        cond = self.condition(img)
        R1 = rot_gt

        n = R1.shape[0]
        R0 = matrix_flow.random_rotation_matrices(n).to(R1.device)
        t = torch.rand(n, device=R1.device)

        Rt = matrix_flow.geodesic_interpolate(R0, R1, t)
        target = matrix_flow.relative_log(R0, R1)
        pred = self.velocity(Rt, t, cond)
        return (pred - target).pow(2).sum(-1).mean()

    def _medoid(self, mats):
        b, k = mats.shape[:2]
        a = mats.unsqueeze(2).expand(b, k, k, 3, 3).reshape(-1, 3, 3)
        c = mats.unsqueeze(1).expand(b, k, k, 3, 3).reshape(-1, 3, 3)
        dist = matrix_flow.geodesic_distance(a, c).view(b, k, k)
        idx = dist.sum(-1).argmin(-1)
        return mats[torch.arange(b, device=mats.device), idx]

    @torch.no_grad()
    def predict(self, x, *, n_samples: int = 1, steps: int = 20):
        b = x.shape[0]
        n_samples = max(1, int(n_samples))

        cond = self.condition(x)
        if n_samples > 1:
            cond = cond.repeat_interleave(n_samples, dim=0)

        n = b * n_samples
        R = matrix_flow.random_rotation_matrices(n).to(x.device)

        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n,), i * dt, device=x.device)
            v = self.velocity(R, t, cond)
            R = R @ matrix_flow.exp_map(dt * v)

        if n_samples > 1:
            R = self._medoid(R.view(b, n_samples, 3, 3))

        return R
