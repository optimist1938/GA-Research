import torch
import torch.nn as nn
import torchvision

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