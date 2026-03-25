# Backbone (https://github.com/ParaMind2025/CAN)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import e3nn
from e3nn import o3

from clifford.algebra.cliffordalgebra import CliffordAlgebra
from src.model import TralaleroTralala
from image2sphere.models import SpatialS2Projector, HarmonicS2Features, SO3Convolution
from image2sphere import so3_utils


# Stochaistic depth from the article
def _drop_path(x, drop_prob: float, training: bool):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_(keep_prob) / keep_prob
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# Geometric product using sparce rolling
class CliffordInteraction(nn.Module):
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=(1, 2)):
        super().__init__()
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.act = nn.SiLU()
        self.shifts = [s for s in shifts if s < dim]
        branch_dim = dim * len(self.shifts)
        cat_dim = branch_dim * 2 if cli_mode == 'full' else branch_dim
        self.proj = nn.Conv2d(cat_dim, dim, kernel_size=1)

    def forward(self, z1, z2):
        C = z2 - z1 if self.ctx_mode == 'diff' else z2
        feats = []
        for s in self.shifts:
            C_s = torch.roll(C, shifts=s, dims=1)
            if self.cli_mode in ('wedge', 'full'):
                feats.append(z1 * C_s - C * torch.roll(z1, shifts=s, dims=1))
            if self.cli_mode in ('inner', 'full'):
                feats.append(self.act(z1 * C_s))
        return self.proj(torch.cat(feats, dim=1))

# The core block: computing geometric product between H and the "neighbour" vector C
class CliffordAlgebraBlock(nn.Module):
    def __init__(self, dim, cli_mode='full', ctx_mode='diff', shifts=(1, 2),
                 drop_path=0.1, init_values=1e-5):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.get_state = nn.Conv2d(dim, dim, kernel_size=1)
        self.get_context_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.clifford_interaction = CliffordInteraction(dim, cli_mode, ctx_mode, shifts)
        self.gate_fc = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_values))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_ln = self.norm(x)
        z_state = self.get_state(x_ln)
        z_ctx = self.get_context_local(x_ln)
        g_feat = self.clifford_interaction(z_state, z_ctx)
        gate = torch.sigmoid(self.gate_fc(torch.cat([x_ln, g_feat], dim=1)))
        x_mixed = F.silu(x_ln) + gate * g_feat
        return shortcut + self.drop_path(self.gamma * x_mixed)


# Represents small groups of pixels as vectors of a high dimension (embedding dim)
class GeometricStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        if patch_size == 2:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.SiLU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


# The complete backbone, adapted from CliffordNet: Image (B, 3, 224, 224) -> Geometric Stem (B, 128, 56, 56) ->
# -> Clifford Algebra Block x number_of_stacked_blocks -> (B, 128, 56, 56) -> LayerNorm -> (B, 128, 56, 56) - feature map
class CliffordNetBackbone(nn.Module):

    def __init__(self, embed_dim=128, patch_size=4, cli_mode='full', ctx_mode='diff',
                 shifts=(1, 2), depth=12, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = GeometricStem(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            CliffordAlgebraBlock(dim=embed_dim, cli_mode=cli_mode, ctx_mode=ctx_mode,
                                 shifts=shifts, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = LayerNorm2d(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)



# Naive version with Tralalero Head (naive projections to CL(4, 1) as it was in the first version)
class CliffordNetCompetitor(nn.Module):
    def __init__(self, algebra, embed_dim=128, depth=12, shifts=(1, 2), drop_path_rate=0.1):
        super().__init__()
        self.backbone = CliffordNetBackbone(
            embed_dim=embed_dim,
            patch_size=4,
            cli_mode='full',
            ctx_mode='diff',
            shifts=shifts,
            depth=depth,
            drop_path_rate=drop_path_rate,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projective_matrix = nn.Linear(embed_dim, 32 * 8)
        self.ga_head = TralaleroTralala(algebra, in_features=8)

    def forward(self, x):
        x = self.backbone(x)           # (B, embed_dim, H', W')
        x = self.avgpool(x)            # (B, embed_dim, 1, 1)
        x = x.flatten(1, -1)           # (B, embed_dim)
        x = self.projective_matrix(x)  # (B, 256)
        x = x.reshape(x.shape[0], -1, 32)  # (B, 8, 32)
        x = self.ga_head(x)            # (B, 9, 32)
        x = x[:, :, 0]                 # (B, 9) 
        x = x.reshape(x.shape[0], 3, 3)
        return x



# CliffordNet (insted of ResNet) + I2S spherical convolution logic
class CliffordNetI2S(nn.Module):

    def __init__(
        self,
        num_classes: int = 1,
        embed_dim: int = 128,
        depth: int = 12,
        shifts: tuple = (1, 2),
        drop_path_rate: float = 0.1,
        img_size: int = 224,
        patch_size: int = 4,
        sphere_fdim: int = 512,
        lmax: int = 6,
        f_hidden: int = 8,
        train_grid_rec_level: int = 3,
        train_grid_n_points: int = 4096,
        train_grid_include_gt: bool = False,
        train_grid_mode: str = "healpix",
        eval_grid_rec_level: int = 5,
        eval_use_gradient_ascent: bool = False,
        include_class_label: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_class_label = include_class_label
        self.lmax = lmax
        self.train_grid_rec_level = train_grid_rec_level
        self.train_grid_n_points = train_grid_n_points
        self.train_grid_include_gt = train_grid_include_gt
        self.train_grid_mode = train_grid_mode
        self.eval_grid_rec_level = eval_grid_rec_level
        self.eval_use_gradient_ascent = eval_use_gradient_ascent

        self.encoder = CliffordNetBackbone(
            embed_dim=embed_dim,
            patch_size=patch_size,
            shifts=shifts,
            depth=depth,
            drop_path_rate=drop_path_rate,
        )
        fmap_size = img_size // patch_size
        proj_input_shape = [embed_dim, fmap_size, fmap_size]
        if include_class_label:
            proj_input_shape[0] += num_classes

        self.projector = SpatialS2Projector(proj_input_shape, sphere_fdim, lmax)
        self.feature_sphere = HarmonicS2Features(sphere_fdim, lmax, f_out=f_hidden)

        irreps_in = so3_utils.s2_irreps(lmax)
        self.o3_conv = o3.Linear(
            irreps_in, so3_utils.so3_irreps(lmax),
            f_in=sphere_fdim, f_out=f_hidden, internal_weights=False,
        )
        self.so3_activation = e3nn.nn.SO3Activation(lmax, lmax, torch.relu, 10)
        so3_grid = so3_utils.so3_near_identity_grid()
        self.so3_conv = SO3Convolution(f_hidden, 1, lmax, so3_grid)

        output_xyx = so3_utils.so3_healpix_grid(rec_level=train_grid_rec_level)
        self.register_buffer("output_wigners", so3_utils.flat_wigner(lmax, *output_xyx).transpose(0, 1))
        self.register_buffer("output_rotmats", o3.angles_to_matrix(*output_xyx))

        output_xyx_eval = so3_utils.so3_healpix_grid(rec_level=eval_grid_rec_level)
        try:
            self.eval_wigners = torch.load("eval_rec5.pt")
        except FileNotFoundError:
            self.eval_wigners = so3_utils.flat_wigner(lmax, *output_xyx_eval).transpose(0, 1)
        self.eval_rotmats = o3.angles_to_matrix(*output_xyx_eval)


    def forward(self, x, o):
        x = self.encoder(x)
        if self.include_class_label:
            o_oh = nn.functional.one_hot(o.squeeze(1), num_classes=self.num_classes)
            o_oh_fmap = o_oh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(-2), x.size(-1))
            x = torch.cat((x, o_oh_fmap.float()), dim=1)
        x = self.projector(x)
        weight, _ = self.feature_sphere()
        x = self.o3_conv(x, weight=weight)
        x = self.so3_activation(x)
        x = self.so3_conv(x)
        return x

    def query_train_grid(self, x, gt_rot=None):
        if self.train_grid_mode == "random":
            idx = torch.randint(len(self.output_rotmats), (self.train_grid_n_points,))
            wigners = self.output_wigners[:, idx]
            rotmats = self.output_rotmats[idx]
            if self.train_grid_include_gt:
                try:
                    abg = o3.matrix_to_angles(gt_rot.cpu())
                    wigners[:, :gt_rot.size(0)] = so3_utils.flat_wigner(self.lmax, *abg).transpose(0, 1).to(x.device)
                    rotmats[:gt_rot.size(0)] = gt_rot
                except AssertionError:
                    pass
        else:  
            wigners = self.output_wigners
            rotmats = self.output_rotmats
        return torch.matmul(x, wigners).squeeze(1), rotmats

    def compute_loss(self, img, cls, rot):
        x = self.forward(img, cls)
        grid_signal, rotmats = self.query_train_grid(x, rot)
        rot_id = so3_utils.nearest_rotmat(rot, rotmats)
        loss = nn.CrossEntropyLoss()(grid_signal, rot_id)
        with torch.no_grad():
            pred_id = grid_signal.max(dim=1)[1]
            acc = so3_utils.rotation_error(rot, rotmats[pred_id])
        return loss, acc.cpu().numpy()

    def predict(self, x, o):
        with torch.no_grad():
            fourier = self.forward(x, o)
            probs = torch.matmul(fourier.cpu(), self.eval_wigners).squeeze(1)
            pred_id = probs.max(dim=1)[1]
        return self.eval_rotmats[pred_id]

    @torch.no_grad()
    def compute_probabilities(self, x, o):
        harmonics = self.forward(x, o).cpu()
        probs = torch.matmul(harmonics, self.eval_wigners).squeeze(1)
        return nn.Softmax(dim=1)(probs)
