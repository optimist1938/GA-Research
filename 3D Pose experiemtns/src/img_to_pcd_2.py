import math
from typing import Optional

import torch
import torch.nn as nn
from image2sphere.so3_utils import so3_healpix_grid, nearest_rotmat
from e3nn import o3

_DA_HIDDEN_SIZES = {"small": 384, "base": 768, "large": 1024}


def _infer_hidden_size(config) -> Optional[int]:
    for attr in ("hidden_size", "backbone_config.hidden_size", "encoder_config.hidden_size"):
        try:
            val = config
            for part in attr.split("."):
                val = getattr(val, part)
            if isinstance(val, int):
                return val
        except AttributeError:
            pass
    return None


# --------------------------------------------------------------------------- #
#  Backbone                                                                    #
# --------------------------------------------------------------------------- #

class DepthAnythingV2Backbone(nn.Module):
    """DepthAnythingV2 multi-layer hidden-state extractor.

    Returns concatenated patch-level features at native ViT patch resolution
    (16×16 for 224×224 input with patch_size=14).  No interpolation, no CNN.
    output shape: [B, hidden_size * len(layers), s, s]
    """

    def __init__(
        self,
        model_size: str = "base",
        layers: tuple = (-1, -3, -6, -9),
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.layers = tuple(layers)
        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.config.output_hidden_states = True

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        hidden_size = _infer_hidden_size(self.model.config)
        if hidden_size is None:
            hidden_size = _DA_HIDDEN_SIZES.get(model_size)
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * len(layers) if hidden_size is not None else None

    @staticmethod
    def _to_feature_map(h: torch.Tensor) -> torch.Tensor:
        """[B, N+1, C] → [B, C, s, s]  (removes CLS token, reshapes to square grid)."""
        patches = h[:, 1:]                                   # drop CLS
        b, n, c = patches.shape
        s = int(math.sqrt(n))
        return patches.transpose(1, 2).reshape(b, c, s, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_enabled = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(grad_enabled):
            out = self.model(pixel_values=x, output_hidden_states=True)
        maps = [self._to_feature_map(out.hidden_states[i]) for i in self.layers]
        return torch.cat(maps, dim=1)                        # [B, hidden_size*L, s, s]


# --------------------------------------------------------------------------- #
#  I2P — DepthAnything backbone + token_mlp head (direct regression)          #
# --------------------------------------------------------------------------- #

class I2P(nn.Module):
    """DepthAnythingV2 hidden-state backbone with token-MLP pose head."""

    def __init__(
        self,
        model_size: str = "base",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = DepthAnythingV2Backbone(model_size, freeze=freeze_backbone)

        in_dim = self.backbone.output_dim
        first_layer = (
            nn.Linear(in_dim, 1024) if in_dim is not None else nn.LazyLinear(1024)
        )
        self.token_mlp = nn.Sequential(
            first_layer,
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
        feat = self.backbone(x)                              # [B, C, s, s]
        b, c, h, w = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, s*s, C]
        emb = self.token_mlp(tokens).mean(dim=1)             # [B, 512]
        return self.pose_head(emb).view(-1, 3, 3)


# --------------------------------------------------------------------------- #
#  I2P_IPDF — Implicit-PDF probabilistic model                                #
#  Paper: https://arxiv.org/pdf/2106.05965                                    #
# --------------------------------------------------------------------------- #

class I2P_IPDF(nn.Module):
 

    def __init__(
        self,
        model_size: str = "base",
        freeze_backbone: bool = True,
        rec_level: int = 3,
        n_train_queries: int = 4096,
        pe_freqs: int = 4,
        eval_chunk_size: int = 4096,
        grad_ascent_steps: int = 100,
        grad_ascent_lr: float = 1e-4,
    ):
        super().__init__()
        self.pe_freqs = pe_freqs
        self.n_train_queries = n_train_queries
        self.eval_chunk_size = eval_chunk_size
        self.grad_ascent_steps = grad_ascent_steps
        self.grad_ascent_lr = grad_ascent_lr

        self.backbone = DepthAnythingV2Backbone(model_size, freeze=freeze_backbone)

        in_dim = self.backbone.output_dim
        first_token_layer = (
            nn.Linear(in_dim, 1024) if in_dim is not None else nn.LazyLinear(1024)
        )
        self.token_mlp = nn.Sequential(
            first_token_layer,
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

        img_dim = 512
        # raw 9 elements + sin/cos for each frequency
        rot_pe_dim = 9 + 2 * pe_freqs * 9

        # Efficient first layer: W_img * d + W_rot * q  (IPDF batching trick)
        self.mlp_img_proj = nn.Linear(img_dim, 256, bias=True)
        self.mlp_rot_proj = nn.Linear(rot_pe_dim, 256, bias=False)

        # Remaining 3 layers → scalar  (4 total as in paper)
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
        r = R.reshape(*shape, 9)
        freqs = (2 ** torch.arange(self.pe_freqs, device=R.device, dtype=R.dtype)) * math.pi
        x = r.unsqueeze(-1) * freqs                          # [..., 9, pe_freqs]
        pe = torch.cat([x.sin(), x.cos()], dim=-1)           # [..., 9, 2*pe_freqs]
        pe = pe.reshape(*shape, 9 * 2 * self.pe_freqs)
        return torch.cat([r, pe], dim=-1)                    # [..., 9 + 2*pe_freqs*9]

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
         feat = self.backbone(x)                              # [B, C, s, s]
        b, c, h, w = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, s*s, C]
        return self.token_mlp(tokens).mean(dim=1)            # [B, 512]

    def _score(self, img_emb: torch.Tensor, rot_pe: torch.Tensor) -> torch.Tensor:
  
        if rot_pe.dim() == 3:
            h = self.mlp_img_proj(img_emb).unsqueeze(1) + self.mlp_rot_proj(rot_pe)
        else:
            h = self.mlp_img_proj(img_emb) + self.mlp_rot_proj(rot_pe)
        return self.mlp_body(h).squeeze(-1)

    def _score_chunked(self, img_emb: torch.Tensor, rots: torch.Tensor) -> torch.Tensor:
 
        chunks = []
        for i in range(0, rots.shape[0], self.eval_chunk_size):
            chunk = rots[i : i + self.eval_chunk_size]               # [C, 3, 3]
            rot_pe = self._positional_encode(chunk)                  # [C, pe_dim]
            rot_pe = rot_pe.unsqueeze(0).expand(img_emb.shape[0], -1, -1)
            chunks.append(self._score(img_emb, rot_pe))
        return torch.cat(chunks, dim=1)                              # [B, N]

    @staticmethod
    def _sample_random_so3(n: int, device) -> torch.Tensor:
         M = torch.randn(n, 3, 3, device=device)
        Q, _ = torch.linalg.qr(M)
        # Sign-correct each column via diagonal of R → Haar-uniform on O(3)
        d = torch.diagonal(R, dim1=-2, dim2=-1).sign()        # [n, 3]
        Q = Q * d.unsqueeze(-2)                               # [n, 3, 3]
        # Project O(3) → SO(3): flip last column where det = -1
        dets = torch.det(Q)
        flip = torch.ones(n, 3, device=device)
        flip[:, 2] = dets.sign()
        return Q * flip.unsqueeze(-2)


    @staticmethod
    def _matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        s = torch.sqrt((trace + 1).clamp(min=1e-10)) * 2   # = 4w
        q = torch.stack([
            0.25 * s,
            (R[:, 2, 1] - R[:, 1, 2]) / s,
            (R[:, 0, 2] - R[:, 2, 0]) / s,
            (R[:, 1, 0] - R[:, 0, 1]) / s,
        ], dim=-1)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)

    @staticmethod
    def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
            2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x),
            2*(x*z - w*y),       2*(y*z + w*x),   1 - 2*(x*x + y*y),
        ], dim=-1).reshape(q.shape[0], 3, 3)
        return R

    def _gradient_ascent(
        self, img_emb: torch.Tensor, R_init: torch.Tensor
    ) -> torch.Tensor:
        # predict() runs under @torch.no_grad(), so we must re-enable autograd here.
        q = self._matrix_to_quat(R_init).detach()

        for _ in range(self.grad_ascent_steps):
            q = q.detach().requires_grad_(True)
            with torch.enable_grad():
                R = self._quat_to_matrix(q)
                rot_pe = self._positional_encode(R)
                score = self._score(img_emb, rot_pe).sum()
                # autograd.grad computes gradient only for q,
                # does NOT accumulate gradients into MLP parameters.
                (grad_q,) = torch.autograd.grad(score, q)
            q = (q + self.grad_ascent_lr * grad_q).detach()
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        return self._quat_to_matrix(q)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self._encode_image(x)
        return self._score_chunked(img_emb, self.so3_rotmats_cache)

    def compute_loss(
        self,
        img: torch.Tensor,
        rot_gt: torch.Tensor,
        criterion,                
    ) -> torch.Tensor:
        b = img.shape[0]
        img_emb = self._encode_image(img)

        sampled = self._sample_random_so3(self.n_train_queries - 1, img.device)
        sampled = sampled.unsqueeze(0).expand(b, -1, -1, -1)            # [B, N-1, 3, 3]
        queries = torch.cat([rot_gt.unsqueeze(1), sampled], dim=1)      # [B, N, 3, 3]

        rot_pe = self._positional_encode(queries)                        # [B, N, pe_dim]
        logits = self._score(img_emb, rot_pe)                           # [B, N]

        nll = -(logits[:, 0] - torch.logsumexp(logits, dim=1))
        return nll.mean()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self._encode_image(x)
        logits = self._score_chunked(img_emb, self.so3_rotmats_cache)  # [B, N_grid]
        best_idx = torch.argmax(logits, dim=-1)                         # [B]
        R_init = self.so3_rotmats_cache[best_idx]                       # [B, 3, 3]

        if self.grad_ascent_steps > 0:
            R_init = self._gradient_ascent(img_emb, R_init)

        return R_init

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor) -> torch.Tensor:
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)
