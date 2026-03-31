import torch
import torch.nn as nn

from clifford.algebra.cliffordalgebra import CliffordAlgebra
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.linear import MVLinear



class SmallCNN(nn.Module):
    def __init__(self, out_channels=16, pool=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )
        self._pool = nn.AdaptiveAvgPool2d(1) if pool else None

    def forward(self, x):
        x = self.net(x)
        if self._pool is not None:
            return self._pool(x).squeeze(-1).squeeze(-1) 
        return x  


class MLPHead(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class CliffordHead(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.hidden = hidden

        self.embed   = nn.Linear(in_features, hidden * 4)
        self.gp      = SteerableGeometricProductLayer(self.algebra, hidden)
        self.readout = nn.Linear(hidden * 4, 2)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x).reshape(B, self.hidden, 4)
        h = self.gp(h)
        return self.readout(h.reshape(B, -1))


class CliffordHeadScalar(nn.Module):
    def __init__(self, in_features, hidden, mode='scalar'):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.hidden = hidden
        self.mode = mode 

        self.embed   = nn.Linear(in_features, hidden * 4)
        self.gp      = SteerableGeometricProductLayer(self.algebra, hidden)
        self.readout = nn.Linear(hidden, 2)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x).reshape(B, self.hidden, 4)
        h = self.gp(h)                              
        if self.mode == 'vnorm':
            feats = h[:, :, 1:3].norm(dim=-1)     
        else:
            feats = h[:, :, 0]                    
        return self.readout(feats)


class CliffordHeadSpatial(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.gp      = SteerableGeometricProductLayer(self.algebra, n_channels)
        self.readout = MVLinear(self.algebra, n_channels, 1)

    def forward(self, feat):
        B, C, H, W = feat.shape

        xs = torch.linspace(-1, 1, W, device=feat.device)
        ys = torch.linspace(-1, 1, H, device=feat.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')

        mv = torch.stack([
            feat.mean(dim=(-2, -1)),
            (feat * gx).mean(dim=(-2, -1)),
            (feat * gy).mean(dim=(-2, -1)),
            torch.zeros(B, C, device=feat.device),
        ], dim=-1)                                    

        mv = self.gp(mv)                               
        mv = self.readout(mv)                        
        return mv[:, 0, 1:3]                     


class CliffordHeadPos(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.gp = SteerableGeometricProductLayer(self.algebra, n_channels)
        self.readout = nn.Linear(n_channels, 2)

    def forward(self, feat):
        B, C, H, W = feat.shape

        xs = torch.linspace(-1, 1, W, device=feat.device)
        ys = torch.linspace(-1, 1, H, device=feat.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')  

        scalars = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        e1 = gx.reshape(H * W)  
        e2 = gy.reshape(H * W)

        mv = torch.zeros(B, H * W, C, 4, device=feat.device)
        mv[:, :, :, 0] = scalars
        mv[:, :, :, 1] = e1[None, :, None]  
        mv[:, :, :, 2] = e2[None, :, None]

        mv = mv.reshape(B * H * W, C, 4)
        mv = self.gp(mv)                                    
        mv = mv.reshape(B, H * W, C, 4)

        mv = mv.mean(dim=1)                                   
        return self.readout(mv[:, :, 1:3].norm(dim=-1))    


class RotationNet(nn.Module):
    def __init__(self, cnn_channels=16, head='clifford', head_hidden=8):
        super().__init__()

        if head in ('clifford_spatial', 'clifford_pos'):
            self.cnn  = SmallCNN(out_channels=cnn_channels, pool=False)
            self.head = CliffordHeadSpatial(cnn_channels) if head == 'clifford_spatial' \
                        else CliffordHeadPos(cnn_channels)
        else:
            self.cnn = SmallCNN(out_channels=cnn_channels, pool=True)

            if head == 'clifford':
                self.head = CliffordHead(cnn_channels, hidden=head_hidden)

            elif head == 'clifford_scalar':
                self.head = CliffordHeadScalar(cnn_channels, hidden=head_hidden, mode='scalar')

            elif head == 'clifford_vnorm':
                self.head = CliffordHeadScalar(cnn_channels, hidden=head_hidden, mode='vnorm')

            elif head == 'mlp':
                clf_params = sum(
                    p.numel() for p in CliffordHead(cnn_channels, head_hidden).parameters()
                )
                mlp_hidden = max(2, round((clf_params - 2) / (cnn_channels + 3)))
                self.head = MLPHead(cnn_channels, hidden=mlp_hidden)

    def forward(self, x):
        return self.head(self.cnn(x))
