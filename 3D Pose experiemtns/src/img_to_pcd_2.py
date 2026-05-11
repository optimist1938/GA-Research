import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline


class PointCloudProcessor(nn.Module):
    def __init__(self, model_size: str = "base", device: Optional[str] = None, freeze: bool = True, layers=(-3, -6, -9, -12)):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        self.pipe = pipeline("depth-estimation", model=name, device=self.device)
        self.model, self.image_processor = self.pipe.model, self.pipe.image_processor
        self.model.config.output_hidden_states = True
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _tok2map(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        t = t[:, 1:]
        s = int(math.sqrt(t.shape[1]))
        return F.interpolate(t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s), (h, w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        with torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            y = self.model(x)
        feat = torch.cat([self._tok2map(y.hidden_states[i], h, w) for i in self.layers], dim=1)
        return y.predicted_depth, feat


class I2P(nn.Module):
    def __init__(self, model_size="base", n_points=2048, hidden_dim=256, device: Optional[str] = None, freeze_backbone=True):
        super().__init__()
        dims = {"small": 384, "base": 768, "large": 1024}
        c = dims[model_size]
        self.n_points = n_points
        self.backbone = PointCloudProcessor(model_size, device=device, freeze=freeze_backbone)
        self.fuse = nn.Sequential(nn.Conv2d(c * 4, 128, 1), nn.GELU())
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

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        depth, feat = self.backbone(x)
        feat = self.fuse(feat).permute(0, 2, 3, 1)

        u, v = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        grid = torch.stack((v, u), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

        pts = torch.cat((grid, depth.unsqueeze(-1)), dim=-1).reshape(b, -1, 3)
        feat = feat.reshape(b, -1, feat.shape[-1])
        pts = torch.cat((pts, feat), dim=-1)

        if pts.shape[1] > self.n_points:
            idx = torch.randperm(pts.shape[1], device=x.device)[:self.n_points]
            pts = pts[:, idx]

        z = self.enc(pts).max(dim=1).values
        R = self.head(z).view(-1, 3, 3)
        return R
