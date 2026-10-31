import numpy as np
import torch
from pose3d.datasets.symsol import Symsol, _read_png
from pose3d.config import parse_args
from test_symsol import _fake_symsol

def test_shared_pixels_and_nc_labels(tmp_path, monkeypatch):
    _fake_symsol(tmp_path, n_views=3)
    monkeypatch.setenv("SYMSOL_SHARED_CACHE", str(tmp_path / "cache"))
    a = Symsol(str(tmp_path), train=True, num_views=3, train_equivalents=True, workers=2)
    expected = torch.from_numpy(_read_png(tmp_path / "symsol/train/images/tet_00000.png").copy()).permute(2, 0, 1)
    assert torch.equal(a.images[0], expected)
    assert torch.equal(a[0]["img"], expected.float() / 255)
    assert "rots_eq" in a[0] and "n_eq" in a[0]
    monkeypatch.setattr("pose3d.datasets.symsol._read_png", lambda _: (_ for _ in ()).throw(AssertionError("cache missed")))
    b = Symsol(str(tmp_path), train=True, num_views=3, train_equivalents=True, workers=2)
    assert torch.equal(a.images, b.images)

def test_nc_and_mre_launch_flags():
    cfg = parse_args(["--path_to_datasets", "/tmp", "--model", "dit_flow", "--dit_sphere_tokens",
                      "--dit_pose_lmax", "6", "--symsol_nearest_coupling", "--no-symsol_likelihood",
                      "--ddp", "--sync_bn", "--num_gpus", "2"])
    assert cfg.data.symsol_nearest_coupling and not cfg.data.symsol_likelihood
    assert cfg.flow.dit_pose_lmax == 6 and cfg.distributed.num_gpus == 2 and cfg.distributed.sync_bn
