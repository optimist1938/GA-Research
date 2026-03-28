# ViG image→graph (https://arxiv.org/pdf/2206.00272) 
# Clifford Group Equivariant GNN (https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks)

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifford.algebra.cliffordalgebra import CliffordAlgebra
from clifford.models.modules.linear import MVLinear
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.mvlayernorm import MVLayerNorm
from clifford.models.modules.gp import SteerableGeometricProductLayer



class VigStem(nn.Module):
    def __init__(self, embed_dim=192, patch_size=4, img_size=32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim, img_size // patch_size, img_size // patch_size]),
        )
        self.fmap_size = img_size // patch_size

    def forward(self, x):
        return self.convs(x)



def build_knn_graph(H, W, k):
    N = H * W
    y = torch.arange(H).float()
    x = torch.arange(W).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)        
    coords_norm = coords / torch.tensor([W - 1, H - 1]) * 2 - 1

    dist = torch.cdist(coords_norm.unsqueeze(0), coords_norm.unsqueeze(0)).squeeze(0)
    dist.fill_diagonal_(float('inf'))
    knn_idx = dist.topk(k, dim=-1, largest=False).indices               

    rows = torch.arange(N).unsqueeze(1).expand(N, k).reshape(-1)      
    cols = knn_idx.reshape(-1)

    return rows, cols, coords_norm                                        


class CEMLP(nn.Module):
    def __init__(self, algebra, n_features, n_layers=2):
        super().__init__()
        self.layers = nn.Sequential(*[
            SteerableGeometricProductLayer(algebra, n_features)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        return self.layers(x)


class EGCL(nn.Module):
    def __init__(self, algebra, in_features, out_features, residual=True):
        super().__init__()
        self.residual = residual
        # MVLinear handles dimension changes; CEMLP only does grade mixing
        self.edge_proj = MVLinear(algebra, in_features, out_features)
        self.edge_act = MVSiLU(algebra, in_features, out_features)
        self.node_proj = MVLinear(algebra, in_features + out_features, out_features)
        self.node_gp = SteerableGeometricProductLayer(algebra, out_features)
        self.node_act = MVSiLU(algebra, out_features)

    def forward(self, h, edge_index):
        rows, cols = edge_index
        h_msg = self.edge_act(self.edge_proj(h[rows] - h[cols]))
        N = h.shape[0]
        agg = torch.zeros(N, *h_msg.shape[1:], device=h.device)
        count = torch.zeros(N, 1, 1, device=h.device)
        agg.scatter_add_(0, rows.unsqueeze(-1).unsqueeze(-1).expand_as(h_msg), h_msg)
        count.scatter_add_(0, rows.unsqueeze(-1).unsqueeze(-1).expand(len(rows), 1, 1),
                           torch.ones(len(rows), 1, 1, device=h.device))
        node_update = self.node_proj(torch.cat([h, agg], dim=1))
        out = self.node_act(self.node_gp(node_update))
        if self.residual:
            out = h + out
        return out


class VigCGENNClassifier(nn.Module):

    NUM_CLASSES = 10

    def __init__(self, embed_dim=192, n_layers=6, k=5, hidden_dim=64,
                 patch_size=4, img_size=32, drop_path_rate=0.1):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.algebra_dim = 2 ** 2
        self.embed_dim = embed_dim
        self.k = k

        self.stem = VigStem(embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)
        self.fmap_size = self.stem.fmap_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.fmap_size, self.fmap_size))

        rows, cols, coords_norm = build_knn_graph(self.fmap_size, self.fmap_size, k)
        self.register_buffer("graph_rows", rows)         
        self.register_buffer("graph_cols", cols)         
        self.register_buffer("coords_norm", coords_norm) 

        self.input_proj = nn.Sequential(
            MVLinear(self.algebra, embed_dim + 1, hidden_dim, subspaces=False), 
            SteerableGeometricProductLayer(self.algebra, hidden_dim)
        ) # downsampling + performing blade mixing operation 

        self.layers = nn.ModuleList([
            EGCL(self.algebra, hidden_dim, hidden_dim, residual=(i > 0))
            for i in range(n_layers)
        ])

        self.output_proj = MVLinear(self.algebra, hidden_dim, embed_dim, subspaces=False)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, self.NUM_CLASSES)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        N = self.fmap_size * self.fmap_size

        x = self.stem(x) + self.pos_embed                            

        offset = torch.arange(B, device=x.device).unsqueeze(1) * N   
        rows_b = (self.graph_rows.unsqueeze(0) + offset).reshape(-1)
        cols_b = (self.graph_cols.unsqueeze(0) + offset).reshape(-1)
        edge_index = torch.stack([rows_b, cols_b], dim=0)

        feat = x.permute(0, 2, 3, 1).reshape(B * N, self.embed_dim)   

        h = torch.zeros(B * N, self.embed_dim + 1, self.algebra_dim, device=x.device)
        h[:, :self.embed_dim, 0] = feat
        coords_batched = self.coords_norm.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 2)
        h[:, self.embed_dim, 1:3] = coords_batched

        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(h, edge_index)

        h = self.output_proj(h)[..., 0]
        h = self.output_norm(h)
        h = h.reshape(B, N, self.embed_dim).mean(dim=1)
        return self.classifier(h)
