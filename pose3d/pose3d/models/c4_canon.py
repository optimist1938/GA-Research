"""C4 canonicalisation around a trained pose model (variant B): exact 90-degree equivariance, no retraining.

    c(x)   = argmax_k  s( rot90(x, -k) )          s: an 'uprightness' score of one view
    f(x)   = G^c(x) · f0( rot90(x, -c(x)) )        G = R_z(-90 deg), the label change of one turn

`f0` is any trained model (e.g. the 8.95 deg GATr flow) and stays frozen. If `c` is the strict
argmax, turning the image moves it by one (Lean: `orbit_argmax_equivariant`) and `f` is exactly
C4-equivariant (`canonicalize_equivariant`), whatever f0 is. On upright photos, where `c(x) = 0`,
`f(x) = f0(x)` (`canonicalize_of_canonical`): the wrapper costs accuracy only on the images the
scorer gets wrong. The scorer is a logistic regression on f0's own pooled backbone features of the
view (the identity prior of Mondal et al. 2023 is the training target: the upright view is the one
to pick), so the only trained part is 2049 numbers.

Axes as in so2_head.py: torch.rot90(x, 1, dims=(2, 3)) turns the image counter-clockwise on
screen and maps the label R to R_z(-90 deg) R.
"""

import torch
import torch.nn as nn

from pose3d.models.c4_lift import rot90

# R_z(-90 deg): the label change of one counter-clockwise turn of the image.
G_TURN = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class C4Canonicalized(nn.Module):
    def __init__(self, model, scorer_weight=None, scorer_bias=0.0):
        super().__init__()
        self.model = model
        n_feat = model.adapter.backbone.output_shape[0]
        self.scorer = nn.Linear(n_feat, 1)
        with torch.no_grad():
            if scorer_weight is not None:
                self.scorer.weight.copy_(torch.as_tensor(scorer_weight).view(1, -1))
                self.scorer.bias.fill_(float(scorer_bias))

    def view_features(self, x):
        """(B, 4, C): pooled backbone features of rot90(x, -k), k = 0..3 (one batched pass)."""
        b = x.shape[0]
        views = torch.cat([rot90(x, -k) for k in range(4)])
        feats = self.model.adapter.backbone(views).mean((2, 3))
        return feats.view(4, b, -1).transpose(0, 1)

    def canonical_turn(self, x):
        return self.scorer(self.view_features(x)).squeeze(-1).argmax(-1)       # (B,) in 0..3

    def predict(self, x, cls=None, **kw):
        c = self.canonical_turn(x)
        x_c = torch.stack([rot90(xi, -int(ci)) for xi, ci in zip(x, c)])
        rot_c = self.model.predict(x_c, cls, **kw)
        g = torch.stack([torch.linalg.matrix_power(G_TURN, int(ci)) for ci in c]).to(rot_c)
        return g @ rot_c
