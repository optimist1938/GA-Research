"""Graft the DiT geometric-bias field (dit_geobias.py, from the SYMSOL-I geobias kernel's code) into a pose3d tree that
already has --dircloud_nside / --dircloud_grid / --symsol_set 14. The bias gain s is read from POSE3D_GEOBIAS_SCALE
(default 30, the value of the SYMSOL-I run). Every edit must apply exactly once."""
import os, shutil

def edit(path, old, new, count=1):
    s = open(path).read(); assert s.count(old) == count, (path, old, s.count(old)); open(path, "w").write(s.replace(old, new))

def graft(root, geobias_src):
    p = lambda f: os.path.join(root, "pose3d", f)
    shutil.copy(geobias_src, p("models/dit_geobias.py"))
    edit(p("models/dit_geobias.py"), "    def __init__(self, condition_width, *, dim, blocks, heads, mlp_ratio, scale=30.0):\n",
         "    def __init__(self, condition_width, *, dim, blocks, heads, mlp_ratio,\n"
         "                 scale=float(os.environ.get(\"POSE3D_GEOBIAS_SCALE\", \"30\"))):\n")
    edit(p("models/dit_geobias.py"), "import math\n", "import math\nimport os\n")
    # DiTBlock: an optional attention module (the geobias field passes its own)
    edit(p("models/dit_flow.py"), "    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):\n",
         "    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, *, attention=None):\n")
    edit(p("models/dit_flow.py"), "        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)\n",
         "        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True) if attention is None else attention\n")
    # config: the new vector field
    edit(p("config.py"), 'vector_field: Literal["clifford", "gatr", "so3fourier", "rotorcorr"] = ',
         'vector_field: Literal["clifford", "gatr", "so3fourier", "rotorcorr", "dit_geobias"] = ')
    # builder: pass the DiT sizes
    edit(p("models/__init__.py"), "                  dense_linear=fl.gatr_dense_linear, compile=fl.gatr_compile),\n",
         "                  dense_linear=fl.gatr_dense_linear, compile=fl.gatr_compile),\n"
         "        dit=dict(dim=fl.dit_dim, blocks=fl.dit_blocks, heads=fl.dit_heads, mlp_ratio=fl.dit_mlp_ratio),\n")
    cf = p("models/clifford_flow.py")
    edit(cf, "                 gatr_pose_harmonics: int = 0,\n                 gatr_spherical_lmax: int = 0,\n",
         "                 gatr_pose_harmonics: int = 0,\n                 dit: dict = None,\n                 gatr_spherical_lmax: int = 0,\n")
    edit(cf, '        if pose_tokens != "rotor" and (vector_field != "gatr" or mlp_heads):\n',
         '        if vector_field == "dit_geobias" and (pose_tokens != "frame_pin" or cond_tokens != "spherecloud"):\n'
         '            raise ValueError("dit_geobias requires spherecloud and frame_pin")\n'
         '        if pose_tokens != "rotor" and (vector_field not in ("gatr", "dit_geobias") or mlp_heads):\n')
    edit(cf, '        if vector_field not in ("clifford", "gatr", "so3fourier", "rotorcorr"):\n',
         '        if vector_field not in ("clifford", "gatr", "so3fourier", "rotorcorr", "dit_geobias"):\n')
    edit(cf, '        elif pose_tokens == "frame_pin":\n',
         '        elif vector_field == "dit_geobias":\n'
         '            from pose3d.models.dit_geobias import DiTGeometricBiasField\n'
         '            self.vector_field = DiTGeometricBiasField(self.adapter.conv_adapter.width, **(dit or {}))\n'
         '        elif pose_tokens == "frame_pin":\n')
    edit(cf, '        elif vector_field == "gatr":\n            pass   # GATr\'s own initialisation of the read-out\n',
         '        elif vector_field in ("gatr", "dit_geobias"):\n            pass   # GATr\'s / DiT\'s own initialisation of the read-out\n')
