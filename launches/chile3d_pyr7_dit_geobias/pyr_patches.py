# Patch the snapshot: --dircloud_nside and POSE3D_SYMSOL_COS_SCALE (the same edit as the repository's commit; every
# replacement is asserted to apply exactly once, so a different snapshot fails here instead of training something else).
"""Adds --dircloud_nside (HEALPix density of the spherecloud tokens) to a pose3d tree; with cos=True also makes the cosine
attention's start temperature settable (POSE3D_SYMSOL_COS_SCALE). Every edit must apply exactly once."""
import os, sys

def edit(path, old, new, count=1):
    s = open(path).read(); assert s.count(old) == count, (path, old, s.count(old)); open(path, "w").write(s.replace(old, new))

def apply(root, cos=False):
    p = lambda f: os.path.join(root, "pose3d", f)
    edit(p("config.py"), "    dircloud_scalars: int = 64\n",
         "    dircloud_scalars: int = 64\n"
         "    # With cond_tokens=spherecloud: HEALPix nside of the hemisphere the tokens sit on. 4: 88 points, at most 16 per\n"
         "    # ring around the optical axis (azimuthal frequencies below 8); 8: 368 points, up to 32 per ring.\n"
         "    dircloud_nside: int = 4\n")
    edit(p("models/__init__.py"), "        dircloud_scalars=fl.dircloud_scalars,\n",
         "        dircloud_scalars=fl.dircloud_scalars,\n        dircloud_nside=fl.dircloud_nside,\n")
    edit(p("models/clifford_flow.py"), "                 dircloud_scalars: int = 64,\n",
         "                 dircloud_scalars: int = 64,\n                 dircloud_nside: int = 4,\n", count=2)
    edit(p("models/clifford_flow.py"), "SphereCloudHead(backbone_channels, scalars=dircloud_scalars)",
         "SphereCloudHead(backbone_channels, scalars=dircloud_scalars, nside=dircloud_nside)")
    edit(p("models/clifford_flow.py"), "                dircloud_scalars=dircloud_scalars, dircloud_up_token=dircloud_up_token,\n",
         "                dircloud_scalars=dircloud_scalars, dircloud_nside=dircloud_nside, dircloud_up_token=dircloud_up_token,\n")
    edit(p("evaluate.py"), '        dircloud_scalars=saved.get("dircloud_scalars", 64),\n',
         '        dircloud_scalars=saved.get("dircloud_scalars", 64),\n        dircloud_nside=saved.get("dircloud_nside", 4),\n')
    if cos:
        edit(p("models/gatr_denoiser.py"), 'scale=30.0, fused=True)',
             'scale=float(os.environ.get("POSE3D_SYMSOL_COS_SCALE", "30")), fused=True)')




def apply_grid(root):
    p = lambda f: os.path.join(root, "pose3d", f)
    edit(p("config.py"), "    dircloud_nside: int = 4\n",
         "    dircloud_nside: int = 4\n"
         "    # With cond_tokens=spherecloud and a ResNet: side of the feature map the tokens are lifted from. 7: the ResNet's\n"
         "    # own map at 224 px. 14: its last stage runs at stride 1 with dilation 2 (models/encoders.dilate_last_stage; the\n"
         "    # pretrained weights are kept), so the map is 14x14 and the lift's Gaussian is narrowed in proportion.\n"
         "    dircloud_grid: int = 7\n")
    edit(p("models/__init__.py"), "        dircloud_nside=fl.dircloud_nside,\n",
         "        dircloud_nside=fl.dircloud_nside,\n        dircloud_grid=fl.dircloud_grid,\n")
    edit(p("models/clifford_flow.py"), "                 dircloud_nside: int = 4,\n",
         "                 dircloud_nside: int = 4,\n                 dircloud_grid: int = 7,\n", count=2)
    edit(p("models/clifford_flow.py"),
         "        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone,\n"
         "                                      depth_anything_model=depth_anything_model)\n",
         "        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone,\n"
         "                                      depth_anything_model=depth_anything_model)\n"
         "        if cond_tokens == \"spherecloud\" and dircloud_grid != 7:\n"
         "            from pose3d.models.encoders import dilate_last_stage\n"
         "            if dircloud_grid != 14:\n"
         "                raise ValueError(\"dircloud_grid must be 7 or 14\")\n"
         "            dilate_last_stage(self.backbone)\n")
    edit(p("models/clifford_flow.py"), "SphereCloudHead(backbone_channels, scalars=dircloud_scalars, nside=dircloud_nside)",
         "SphereCloudHead(backbone_channels, scalars=dircloud_scalars, nside=dircloud_nside,\n"
         "                                                grid=dircloud_grid)")
    edit(p("models/clifford_flow.py"), "dircloud_nside=dircloud_nside, dircloud_up_token=dircloud_up_token,\n",
         "dircloud_nside=dircloud_nside, dircloud_grid=dircloud_grid,\n                dircloud_up_token=dircloud_up_token,\n")
    edit(p("evaluate.py"), '        dircloud_nside=saved.get("dircloud_nside", 4),\n',
         '        dircloud_nside=saved.get("dircloud_nside", 4),\n        dircloud_grid=saved.get("dircloud_grid", 7),\n')
    edit(p("models/dircloud_head.py"), "    def __init__(self, c_in: int = 2048, scalars: int = 64, nside: int = 4):\n",
         "    def __init__(self, c_in: int = 2048, scalars: int = 64, nside: int = 4, grid: int = 7):\n")
    edit(p("models/dircloud_head.py"), "        self.lift = SphereLift(c_in, c_s=scalars, nside=nside)\n",
         "        # the lift's Gaussian (0.2 for the 7x7 map, in units of half the map) follows the cell size\n"
         "        self.lift = SphereLift(c_in, c_s=scalars, nside=nside, grid=grid, sigma=0.2 * 6 / (grid - 1))\n")
    edit(p("models/so3fourier_head.py"),
         "        if grid % 2 == 0 or grid < 3:\n"
         "            raise ValueError(\"grid must be odd (>= 3) so the centre cell sits on the optical axis\")\n",
         "        if grid < 3:\n"
         "            raise ValueError(\"grid must be >= 3\")\n")
    edit(p("models/so3fourier_head.py"),
         "        off = (torch.arange(grid, dtype=torch.float64) - grid // 2) / (grid // 2)\n",
         "        # cell centres in [-1, 1]; an odd grid has a centre cell on the optical axis, an even one straddles it\n"
         "        off = (torch.arange(grid, dtype=torch.float64) - (grid - 1) / 2) / ((grid - 1) / 2)\n")
    edit(p("models/encoders.py"), "def build_encoder(encoder_type: str, pretrained: bool = False,\n",
         "def dilate_last_stage(encoder):\n"
         "    \"\"\"A torchvision ResNet behind image2sphere's wrapper: run its last stage at stride 1 with dilation 2, as\n"
         "    torchvision's replace_stride_with_dilation does, so a 224 px image gives a 14x14 map instead of 7x7. The\n"
         "    weights are untouched, so pretrained ones stay valid.\"\"\"\n"
         "    net = getattr(encoder, \"encoder\", encoder)             # ImageNetNormalized wraps the ResNet\n"
         "    stage = net.layers[-1]\n"
         "    first = stage[0]\n"
         "    if not hasattr(first, \"conv3\"):\n"
         "        raise ValueError(\"dilate_last_stage expects a bottleneck ResNet (resnet50 / 101 / 152)\")\n"
         "    first.conv2.stride = (1, 1)\n"
         "    first.downsample[0].stride = (1, 1)\n"
         "    for block in list(stage)[1:]:\n"
         "        block.conv2.dilation, block.conv2.padding = (2, 2), (2, 2)\n"
         "    shape = (net.output_shape[0], 14, 14)\n"
         "    net.output_shape = shape\n"
         "    if net is not encoder:\n"
         "        encoder.output_shape = shape\n"
         "    return encoder\n\n\n"
         "def build_encoder(encoder_type: str, pretrained: bool = False,\n")



def apply_chile3d(root):
    """--symsol_set 14: the 7-gonal pyramid with poses uniform on SO(3), from symsol_chile/ (3D SYMSOL-CHILE)."""
    f = os.path.join(root, "pose3d", "datasets/symsol.py"); t = open(f).read()
    assert t.count("SHAPE_SETS = {\n") == 1 and t.count("SET_DIRS = {") == 1, "symsol.py layout changed"
    t = t.replace("SHAPE_SETS = {\n", "SHAPE_SETS = {\n    14: (\"pyr7\",),   # SYMSOL-CHILE 7-gon, 3D poses (set 7 = its bottom view)\n")
    t = t.replace("SET_DIRS = {", "SET_DIRS = {14: \"symsol_chile\", ")
    open(f, "w").write(t)



