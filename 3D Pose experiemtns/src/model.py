# Original code : https://pin.it/5fZhohFry

import torch
import torch.nn as nn
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from image2sphere.models import ResNet


class TralaleroTralala(nn.Module):
    def __init__(self, algebra, in_features=512, hidden_dim=32, out_features=9):
        super().__init__()
        self.fcgp1 = FullyConnectedSteerableGeometricProductLayer(algebra, in_features=in_features, out_features=hidden_dim)
        self.activ = MVSiLU(algebra, hidden_dim)
        self.gp1 = SteerableGeometricProductLayer(algebra, hidden_dim)
        self.gp2 = FullyConnectedSteerableGeometricProductLayer(algebra, hidden_dim, out_features)

    def forward(self, x):
        x = self.fcgp1(x)
        x = self.activ(x)
        x = self.gp1(x)
        x = self.activ(x)
        x = self.gp2(x)
        return x


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.backbone = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projective_matrix = nn.Linear(2048, 32 * 8)
        self.ga_head = TralaleroTralala(algebra, in_features=8)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.projective_matrix(x)
        x = x.reshape(x.shape[0], -1, 32)
        x = self.ga_head(x)
        x = x[:, :, 0]
        x = x.reshape(x.shape[0], 3, 3)
        return x


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_head = nn.Linear(in_features=2048, out_features=9)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.linear_head(x)
        x = x.reshape(x.shape[0], 3, 3)
        return x
