import torch
import torch.nn as nn

from clifford.algebra.cliffordalgebra import CliffordAlgebra
from clifford.models.modules.gp import SteerableGeometricProductLayer



class SmallCNN(nn.Module):
    def __init__(self, out_channels=16):
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
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1).squeeze(-1)  


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

        self.embed   = nn.Linear(in_features, hidden * 4) # dummy embed
        self.gp      = SteerableGeometricProductLayer(self.algebra, hidden)
        self.readout = nn.Linear(hidden * 4, 2)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x).reshape(B, self.hidden, 4)  
        h = self.gp(h)                                  
        return self.readout(h.reshape(B, -1))          



class CliffordHeadScalar(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0))
        self.hidden = hidden

        self.embed   = nn.Linear(in_features, hidden * 4)
        self.gp      = SteerableGeometricProductLayer(self.algebra, hidden)
        self.readout = nn.Linear(hidden, 2)       

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x).reshape(B, self.hidden, 4)
        h = self.gp(h)                           
        scalars = h[:, :, 0]                     
        return self.readout(scalars)


class RotationNet(nn.Module):
    def __init__(self, cnn_channels=16, head='clifford', head_hidden=8):
        super().__init__()
        self.cnn = SmallCNN(out_channels=cnn_channels)

        if head == 'clifford':
            self.head = CliffordHead(cnn_channels, hidden=head_hidden)

        elif head == 'clifford_scalar':
            self.head = CliffordHeadScalar(cnn_channels, hidden=head_hidden)

        elif head == 'mlp':
            clf_params = sum(
                p.numel() for p in CliffordHead(cnn_channels, head_hidden).parameters()
            )
            mlp_hidden = max(2, round((clf_params - 2) / (cnn_channels + 3)))
            self.head = MLPHead(cnn_channels, hidden=mlp_hidden)

    def forward(self, x):
        return self.head(self.cnn(x))
