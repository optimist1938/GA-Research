"""Every `cfg.<group>.<field>` the code reads must exist in the config dataclasses.

A merge once dropped RunConfig.viz_every while engine/trainer.py still read it: the run crashed at the end of its first epoch,
after the data and the model were set up, on a GPU that cannot show a log until the kernel ends. This reads the source instead.
"""
import glob
import os
import re

from pose3d.config import parse_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_config_field_the_code_reads_exists():
    cfg = parse_args(["--path_to_datasets=/nonexistent"])
    groups = [k for k, v in vars(cfg).items() if hasattr(v, "__dataclass_fields__")]
    files = glob.glob(os.path.join(ROOT, "pose3d", "**", "*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "scripts", "*.py"))
    missing = []
    for f in files:
        for g, a in re.findall(r"\bcfg\.(\w+)\.(\w+)", open(f).read()):
            if g in groups and not hasattr(getattr(cfg, g), a):
                missing.append(f"{os.path.relpath(f, ROOT)}: cfg.{g}.{a}")
    assert not missing, missing
