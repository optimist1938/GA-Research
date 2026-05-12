import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from image2sphere.so3_utils import so3_healpix_grid, nearest_rotmat
from e3nn import o3


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
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            ResBlock(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
        )
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
        feats = [
            F.interpolate(
                a(self._tok2map(y.hidden_states[i], h, w)),
                (h, w), mode="bilinear", align_corners=False,
            )
            for a, i in zip(self.adapt, self.layers)
        ]
        return y.predicted_depth, self.fuse(sum(feats))


# --------------------------------------------------------------------------- #
#  I2P — DepthAnything backbone + token_mlp head (direct regression)          #
# --------------------------------------------------------------------------- #

class I2P(nn.Module):
    """DepthAnythingV2 backbone with token-MLP pose head (direct regression)."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        freeze_backbone: bool = True,
        pool_hw: int = 16,
    ):
        super().__init__()
        self.pool_hw = pool_hw
        self.backbone = PointCloudProcessor(model_size, device=device, freeze=freeze_backbone)

        # 128 channels from PointCloudProcessor.fuse
        self.token_mlp = nn.Sequential(
            nn.Linear(128, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        _, feat = self.backbone(x)                              # [B, 128, H, W]
        feat = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw))  # [B, 128, P, P]
        tokens = feat.permute(0, 2, 3, 1).reshape(b, -1, 128)  # [B, P*P, 128]
        embeddings = self.token_mlp(tokens)                     # [B, P*P, 512]
        global_emb = embeddings.mean(dim=1)                     # [B, 512]
        return self.pose_head(global_emb).view(-1, 3, 3)


#  I2P_IPDF — Implicit-PDF probabilistic model                                #
#  Paper: https://arxiv.org/pdf/2106.05965                                    #
# -------------------------------------------------------------------------- #

class I2P_IPDF(nn.Module):

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        freeze_backbone: bool = True,
        pool_hw: int = 16,
        rec_level: int = 2,
        n_train_queries: int = 2048,
        pe_freqs: int = 4,
    ):
        super().__init__()
        self.pool_hw = pool_hw
        self.pe_freqs = pe_freqs
        self.n_train_queries = n_train_queries

        self.backbone = PointCloudProcessor(model_size, device=device, freeze=freeze_backbone)

        self.token_mlp = nn.Sequential(
            nn.Linear(128, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

        img_dim = 512
        rot_pe_dim = 2 * pe_freqs * 9

        # Efficient first layer: W_img * d + W_rot * q (IPDF batching trick)
        self.mlp_img_proj = nn.Linear(img_dim, 256, bias=True)
        self.mlp_rot_proj = nn.Linear(rot_pe_dim, 256, bias=False)

        # Remaining 3 hidden layers + scalar output (4 layers total, as in paper)
        self.mlp_body = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        xyx = so3_healpix_grid(rec_level=rec_level)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*xyx), persistent=False)

    # ------------------------------------------------------------------ #

    def _positional_encode(self, R: torch.Tensor) -> torch.Tensor:
        shape = R.shape[:-2]
        r = R.reshape(*shape, 9)                                          # [..., 9]
        freqs = (2 ** torch.arange(self.pe_freqs, device=R.device, dtype=R.dtype)) * math.pi
        x = r.unsqueeze(-1) * freqs                                       # [..., 9, pe_freqs]
        pe = torch.cat([x.sin(), x.cos()], dim=-1)                        # [..., 9, 2*pe_freqs]
        return pe.reshape(*shape, 9 * 2 * self.pe_freqs)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        _, feat = self.backbone(x)                                        # [B, 128, H, W]
        feat = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw))  # [B, 128, P, P]
        tokens = feat.permute(0, 2, 3, 1).reshape(b, -1, 128)            # [B, P*P, 128]
        return self.token_mlp(tokens).mean(dim=1)                         # [B, 512]

    def _score(self, img_emb: torch.Tensor, rot_pe: torch.Tensor) -> torch.Tensor:
        if rot_pe.dim() == 3:
            # [B, 1, 256] + [B, N, 256] = [B, N, 256]
            h = self.mlp_img_proj(img_emb).unsqueeze(1) + self.mlp_rot_proj(rot_pe)
        else:
            h = self.mlp_img_proj(img_emb) + self.mlp_rot_proj(rot_pe)
        return self.mlp_body(h).squeeze(-1)

    @staticmethod
    def _sample_random_so3(n: int, device) -> torch.Tensor:
        M = torch.randn(n, 3, 3, device=device)
        Q, R = torch.linalg.qr(M)
        Q = Q * torch.det(Q).sign().view(n, 1, 1)
        return Q  # [n, 3, 3]

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self._encode_image(x)
        rot_pe = self._positional_encode(self.so3_rotmats_cache)          # [N_grid, pe_dim]
        rot_pe = rot_pe.unsqueeze(0).expand(img_emb.shape[0], -1, -1)    # [B, N_grid, pe_dim]
        return self._score(img_emb, rot_pe)                               # [B, N_grid]

    def compute_loss(
        self,
        img: torch.Tensor,
        rot_gt: torch.Tensor,
        criterion,                  # ignored — loss is NLL, computed here
    ) -> torch.Tensor:
        b = img.shape[0]
        img_emb = self._encode_image(img)

        n_random = self.n_train_queries - 1
        sampled = self._sample_random_so3(n_random, img.device)           # [N-1, 3, 3]
        sampled = sampled.unsqueeze(0).expand(b, -1, -1, -1)             # [B, N-1, 3, 3]
        queries = torch.cat([rot_gt.unsqueeze(1), sampled], dim=1)        # [B, N, 3, 3]

        rot_pe = self._positional_encode(queries)                         # [B, N, pe_dim]
        logits = self._score(img_emb, rot_pe)                             # [B, N]

        # R_GT is at index 0; NLL = -log softmax(logits)[0]
        nll = -(logits[:, 0] - torch.logsumexp(logits, dim=1))
        return nll.mean()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)                                          # [B, N_grid]
        idx = torch.argmax(logits, dim=-1)                                # [B]
        return self.so3_rotmats_cache[idx]                                # [B, 3, 3]

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor) -> torch.Tensor:
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)
