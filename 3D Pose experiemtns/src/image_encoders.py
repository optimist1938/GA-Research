import torch
import torch.nn as nn


class GAEncoder(nn.Module):
  """SO(2)-SO(3) equivariant geometric algebra encoder.

  For each pixel with channels (R, G, B) and coordinates (x, y), outputs
  six coefficients corresponding to basis blades:
  [e, e123, e1, e2, e13, e23].
  """

  def __init__(self):
    super().__init__()
    self.rgb_to_e = nn.Linear(3, 1, bias=False)
    self.rgb_to_e123 = nn.Linear(3, 1, bias=False)

    self.alpha_1 = nn.Parameter(torch.tensor(1.0))
    self.beta_1 = nn.Parameter(torch.tensor(0.0))
    self.alpha_2 = nn.Parameter(torch.tensor(1.0))
    self.beta_2 = nn.Parameter(torch.tensor(0.0))

    self.output_shape = (6, 224, 224)

  def _coords(self, height: int, width: int, device, dtype):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy

  def forward(self, x):
    b, _, h, w = x.shape
    rgb = x.permute(0, 2, 3, 1).reshape(-1, 3)

    e = self.rgb_to_e(rgb).reshape(b, h, w)
    e123 = self.rgb_to_e123(rgb).reshape(b, h, w)

    xx, yy = self._coords(h, w, x.device, x.dtype)
    xx = xx.unsqueeze(0).expand(b, -1, -1)
    yy = yy.unsqueeze(0).expand(b, -1, -1)

    e1 = self.alpha_1 * xx + self.beta_1 * yy
    e2 = self.alpha_1 * yy - self.beta_1 * xx
    e13 = self.alpha_2 * xx + self.beta_2 * yy
    e23 = self.alpha_2 * yy - self.beta_2 * xx

    out = torch.stack([e, e123, e1, e2, e13, e23], dim=1)
    self.output_shape = tuple(out.shape[1:])
    return out


class GAEncoderCanonical(nn.Module):
  """Geometric algebra encoder with canonical basis ordering.

  For each pixel with channels (R, G, B) and coordinates (x, y), outputs
  eight coefficients in this basis order:
  [e, e1, e2, e3, e12, e13, e23, e123].

  e12 is initialized as zero, and e3 is also zero because the image provides
  only 2D spatial coordinates.
  """

  def __init__(self):
    super().__init__()
    self.rgb_to_e = nn.Linear(3, 1, bias=False)
    self.rgb_to_e123 = nn.Linear(3, 1, bias=False)

    self.alpha_1 = nn.Parameter(torch.tensor(1.0))
    self.beta_1 = nn.Parameter(torch.tensor(0.0))
    self.alpha_2 = nn.Parameter(torch.tensor(1.0))
    self.beta_2 = nn.Parameter(torch.tensor(0.0))

    self.output_shape = (8, 224, 224)

  def _coords(self, height: int, width: int, device, dtype):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy

  def forward(self, x):
    b, _, h, w = x.shape
    rgb = x.permute(0, 2, 3, 1).reshape(-1, 3)

    e = self.rgb_to_e(rgb).reshape(b, h, w)
    e123 = self.rgb_to_e123(rgb).reshape(b, h, w)

    xx, yy = self._coords(h, w, x.device, x.dtype)
    xx = xx.unsqueeze(0).expand(b, -1, -1)
    yy = yy.unsqueeze(0).expand(b, -1, -1)

    e1 = self.alpha_1 * xx + self.beta_1 * yy
    e2 = self.alpha_1 * yy - self.beta_1 * xx
    e13 = self.alpha_2 * xx + self.beta_2 * yy
    e23 = self.alpha_2 * yy - self.beta_2 * xx

    e3 = torch.zeros_like(e)
    e12 = torch.zeros_like(e)

    out = torch.stack([e, e1, e2, e3, e12, e13, e23, e123], dim=1)
    self.output_shape = tuple(out.shape[1:])
    return out

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetNormalized(nn.Module):
  '''Applies ImageNet mean/std before handing the image to a pretrained backbone.

  Pascal3D delivers images in [0, 1], which is not what torchvision's pretrained
  weights were trained on, so the backbone needs the standard normalization to
  make use of them.
  '''
  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder
    self.output_shape = encoder.output_shape
    self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
    self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

  def forward(self, x):
    return self.encoder((x - self.mean) / self.std)


class ImageEncoder(nn.Module):
  '''Define an image encoding network to process image into dense feature map

  Any standard convolutional network or vision transformer could be used here. 
  In the paper, we use ResNet50 pretrained on ImageNet1K for a fair comparison to
  the baselines.  Here, we show an example using a pretrained SWIN Transformer.

  When using a model from torchvision, make sure to remove the head so the output
  is a feature map, not a feature vector
  '''
  def __init__(self):
    super().__init__()
    import torchvision
    self.layers = torchvision.models.swin_v2_t(weights="DEFAULT")

    # last three modules in swin are avgpool,flatten,linear so change to Identity
    self.layers.avgpool = nn.Identity()
    self.layers.flatten = nn.Identity()
    self.layers.head = nn.Identity()

    # we will need shape of feature map for later
    dummy_input = torch.zeros((1, 3, 224, 224))
    self.output_shape = self(dummy_input).shape[1:]
  
  def forward(self, x):
    return self.layers(x)


_RESNET_SIZES = {"resnet": 50, "resnet50": 50, "resnet101": 101}


def is_resnet(encoder_type: str) -> bool:
  return encoder_type in _RESNET_SIZES


def build_encoder(encoder_type: str, pretrained: bool = False):
  if encoder_type in _RESNET_SIZES:
    from image2sphere.models import ResNet
    encoder = ResNet(size=_RESNET_SIZES[encoder_type], pretrained=pretrained)
    return ImageNetNormalized(encoder) if pretrained else encoder
  if encoder_type == "ga":
    return GAEncoder()
  if encoder_type == "ga_canonical":
    return GAEncoderCanonical()
  raise ValueError(f"Unknown encoder type: {encoder_type}")
