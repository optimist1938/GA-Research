"""The published Image2Sphere model wrapped for this repo's training harness."""

import torch
import torch.nn as nn

from pose3d.models.encoders import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEPTH_ANYTHING_DEFAULT,
    build_encoder,
    freeze_encoder,
    is_dense_backbone,
)


class I2SReal(nn.Module):
    """The published Image2Sphere model, adapted to this repo's training harness.

    `I2S` above is not Image2Sphere: it average-pools the feature map before its
    GA head, discarding the spatial structure that the orthographic S2 projection
    exists to exploit. This wraps the real thing from `image2sphere.predictor` so
    a baseline can be trained under the same dataloader, schedule, and metric as
    the models it is meant to be compared against.

    Three seams need adapting:
      * upstream `compute_loss(img, cls, rot)` takes the class label second and
        returns `(loss, acc)`; the harness calls `compute_loss(img, rot, criterion)`
        and wants a scalar back.
      * upstream `forward` always takes a class tensor, even though
        `include_class_label` is off for PASCAL3D+. A zero tensor stands in, and
        `InMemoryDataset` drops `cls` anyway.
      * upstream `predict` evaluates on CPU and returns CPU rotations, because
        `eval_wigners` is a plain attribute that never follows the model to GPU, and
        builds the whole Wigner matrix of the evaluation grid in `__init__` (level 5,
        the paper's 1.875 deg grid, is 455 x 2.36M floats: 4.3 GB, several times that
        while building). Here the grid is built lazily, in chunks, on the device that
        scores, and the argmax runs there against the harness's CUDA ground truth.

    `grid_logits` exposes the scores over that grid, so the SYMSOL log-likelihood is
    read off the same histogram upstream's `evaluate_ll` uses (engine/grid_model_likelihood.py).
    """

    def __init__(
        self,
        encoder_type: str = "resnet101",
        pretrained_backbone: bool = True,
        lmax: int = 6,
        rec_level: int = 3,
        eval_rec_level: int = 3,
        num_classes: int = 12,
        sphere_fdim: int = 512,
        normalize_input: bool = True,
        depth_anything_model: str = DEPTH_ANYTHING_DEFAULT,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        import image2sphere.predictor as _predictor
        from image2sphere.predictor import I2S as UpstreamI2S

        if not is_dense_backbone(encoder_type):
            raise ValueError(
                f"I2SReal expects a dense backbone, got {encoder_type!r}"
            )

        injected = None
        if encoder_type == "depth_anything":
            # BaseSO3Predictor builds its own ResNet from a string, so a non-resnet
            # backbone has to be swapped in at that call. build_encoder already
            # wraps it in ImageNetNormalized, so _prep must not normalize again.
            injected = build_encoder(encoder_type, pretrained=pretrained_backbone,
                                     depth_anything_model=depth_anything_model)
            # build_encoder only wraps in ImageNetNormalized when pretrained.
            normalize_input = not pretrained_backbone
            # The string is unused once ResNet is patched, but BaseSO3Predictor
            # still regex-parses a digit out of it before the call.
            encoder = "resnet50_pretrained"
        else:
            size = 101 if encoder_type == "resnet101" else 50
            encoder = f"resnet{size}" + ("_pretrained" if pretrained_backbone else "")

        original_resnet = _predictor.ResNet
        if injected is not None:
            _predictor.ResNet = lambda *args, **kwargs: injected
        try:
            self.net = UpstreamI2S(
                num_classes=num_classes,
                encoder=encoder,
                sphere_fdim=sphere_fdim,
                lmax=lmax,
                train_grid_rec_level=rec_level,
                train_grid_mode="healpix",
                eval_grid_rec_level=1,   # a stand-in; the real grid is built lazily (see _eval_grid)
                eval_use_gradient_ascent=False,
                include_class_label=False,
            )
        finally:
            _predictor.ResNet = original_resnet

        self.frozen_backbone = bool(freeze_backbone)
        if self.frozen_backbone:
            if not pretrained_backbone:
                raise ValueError(
                    "freeze_backbone with a randomly initialised encoder trains a "
                    "head on noise; pass pretrained_backbone as well."
                )
            # The image never requires grad and neither do these parameters, so the
            # encoder's output detaches on its own -- no autograd graph is built
            # through it and no explicit no_grad is needed.
            freeze_encoder(self.net.encoder)

        # Upstream's level-1 stand-in grid is dropped; the evaluation grid is built on first use.
        del self.net.eval_wigners
        del self.net.eval_rotmats
        self.lmax = lmax
        self.eval_rec_level = eval_rec_level
        self._grid = None

        # Pascal3D hands over [0, 1] images. Upstream feeds those straight to an
        # ImageNet-pretrained ResNet, but every other pretrained path in this repo
        # normalizes first (see ImageNetNormalized), so the baseline is fed the
        # same way as the models it is being compared against.
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "frozen_backbone", False):
            self.net.encoder.eval()
        return self

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std if self.normalize_input else img

    def _cls(self, img: torch.Tensor) -> torch.Tensor:
        # include_class_label is off, so this is only shape-compatibility.
        return torch.zeros(img.shape[0], 1, dtype=torch.long, device=img.device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        '''Returns the SO(3) Fourier coefficients, matching upstream's forward.'''
        return self.net(self._prep(img), self._cls(img))

    def compute_loss(self, img: torch.Tensor, rot_gt: torch.Tensor, criterion=None) -> torch.Tensor:
        # criterion is ignored: upstream builds its own CrossEntropyLoss over the
        # training grid, and reproducing I2S means keeping that. So --loss and
        # --label_smoothing have no effect on this model.
        loss, _acc = self.net.compute_loss(self._prep(img), self._cls(img), rot_gt)
        return loss

    @torch.no_grad()
    def _eval_grid(self, device):
        """(Wigner matrix (M, Q), rotations (Q, 3, 3)) of the SO(3) HEALPix evaluation grid, built in chunks on
        `device` the first time and cached. Level 3 (upstream's default) is 36,864 rotations, level 5 (the
        paper's evaluation) 2,359,296: a 4.3 GB float32 matrix, which only fits built chunk by chunk."""
        device = torch.device(device)
        if self._grid is None or self._grid[0].device != device:
            from e3nn import o3
            from image2sphere import so3_utils

            a, b, g = (t.to(device) for t in so3_utils.so3_healpix_grid(rec_level=self.eval_rec_level))
            q = a.numel()
            m = sum((2 * l + 1) ** 2 for l in range(self.lmax + 1))
            wig = torch.empty(m, q, device=device)
            for i in range(0, q, 65536):
                wig[:, i:i + 65536] = so3_utils.flat_wigner(self.lmax, a[i:i + 65536], b[i:i + 65536],
                                                            g[i:i + 65536]).T
            self._grid = (wig, o3.angles_to_matrix(a, b, g))
        return self._grid

    @torch.no_grad()
    def grid_logits(self, img: torch.Tensor) -> torch.Tensor:
        """Unnormalised log-probabilities of every evaluation-grid rotation, (B, Q): the softmax over them is
        upstream's `compute_probabilities`, the histogram its SYMSOL log-likelihood is read from."""
        wig, _ = self._eval_grid(img.device)
        return torch.matmul(self.forward(img).squeeze(1).float(), wig)

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        _, rots = self._eval_grid(img.device)
        return rots[self.grid_logits(img).argmax(dim=1)]
