from cliffod.models.modules.linear import MVLinear
from cliffod.models.modules.gp import SteerableGeometricProductLayer
from cliffod.models.modules.mvlayernorm import MVLayerNorm
from clifford.models.modules.mvsilu import MVSiLU

import torch.nn as nn 

class CGEBlock(nn.Module):
    def __init__(self, algebra, in_features, out_features):
        super().__init__()
        print(in_features)

        self.layers = nn.Sequential(
                SteerableGeometricProductLayer(algebra, out_features),
                MVLayerNorm(algebra, out_features),
            )
    def forward(self, input):
        # [batch_size, in_features, 2**d] -> [batch_size, out_features, 2**d]
        return self.layers(input)
    

class CGEMLP(nn.Module):
    def __init__(self, algebra, in_features, hidden_features, out_features, n_layers=2):
        super().__init__()

        layers = []
        for i in range(n_layers - 1):
            layers.append(
                CGEBlock(algebra, in_features, hidden_features)
            )
            in_features = hidden_features
        layers.append(
            CGEBlock(algebra, hidden_features, out_features)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
    

class CIFAR10_model(nn.Module):
    def __init__(self, ca, in_channels, hidden_channels, out_classes):
        super().__init__()
        self.ca = ca 
        self.mv_size = 4
        self.pixel_mixer = MVLinear(ca, in_features=3, out_features=1)
        self.qtgp = SteerableGeometricProductLayer(ca, features=hidden_channels)
        self.clifford_head = MVLinear(ca, in_features=hidden_channels, out_features=out_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        colors = x[..., :3] # [batch_size, 1024, 3]
        coords = x[..., 3:] # [batch_size, 1024, 2]

        scalars = self.ca.embed_grade(colors.unsqueeze(-1), 0) # [batch_size, 1024, 1, 3, 4]

        vectors = self.ca.embed_grade(coords, 1).unsqueeze(2) # [batch_size, 1024, 1, 4], broadcasting
        x_mv = scalars + vectors # [batch_size, 1024, 3, 4]
        x_mv = x_mv.view(-1, 3, self.mv_size) # prepare for channel mixing
        x_mv = self.pixel_mixer(x_mv)
        x_global = x_mv.view(batch_size, 1024, self.mv_size) # in_channels = all pixels in the image 
        x_out = self.qtgp(x_global)
        logits_mv = self.clifford_head(x_out) # projecting to out_chan = out_classes
        return logits_mv[..., 1]

