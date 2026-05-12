import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PointCloudProcessor(nn.Module):
    def __init__(self, model_size="base", device: Optional[str] = None, freeze=True, layers=(-1, -3, -6, -9)):
        super().__init__()
        self.layers = layers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dim = {"small": 384, "base": 768, "large": 1024}[model_size]
        name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        p = pipeline("depth-estimation", model=name, device=self.device)
        self.model = p.model
        self.model.config.output_hidden_states = True
        self.model.eval()
        self.adapt = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, 128, 1), nn.GELU(), ResBlock(128))
            for _ in layers
        ])
        self.fuse = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), ResBlock(128), nn.Conv2d(128, 128, 3, padding=1), nn.GELU())
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _tok2map(t, h, w):
        t = t[:, 1:]
        s = int(math.sqrt(t.shape[1]))
        return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)

    def forward(self, x):
        h, w = x.shape[-2:]
        with (torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad()):
            y = self.model(x)
        feats = [F.interpolate(a(self._tok2map(y.hidden_states[i], h, w)), (h, w), mode="bilinear", align_corners=False) for a, i in zip(self.adapt, self.layers)]
        return y.predicted_depth, self.fuse(sum(feats))


class I2P(nn.Module):
    def __init__(self, model_size="base", n_points=2048, hidden_dim=256, device: Optional[str] = None, freeze_backbone=True):
        super().__init__()
        self.n_points = n_points
        self.backbone = PointCloudProcessor(model_size, device=device, freeze=freeze_backbone)
        self.enc = nn.Sequential(
            nn.Linear(3 + 128, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 512), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        depth, feat = self.backbone(x)
        u, v = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        grid = torch.stack((v, u), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        pts = torch.cat((grid, depth.unsqueeze(-1)), dim=-1).reshape(b, -1, 3)
        feat = feat.permute(0, 2, 3, 1).reshape(b, -1, 128)
        pts = torch.cat((pts, feat), dim=-1)
        if pts.shape[1] > self.n_points:
            idx = torch.randperm(pts.shape[1], device=x.device)[:self.n_points]
            pts = pts[:, idx]
        z = self.enc(pts).max(dim=1).values
        R = self.head(z).view(-1, 3, 3)
        return R