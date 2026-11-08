"""Export the reproducible toy-pose splits as NumPy arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import canonical_geometry, make_flow_splits, make_pose_test


def _save(path, arrays):
    np.savez_compressed(path, **{name: tensor.numpy() for name, tensor in arrays.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "pyramid"), default="original")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train, validation = make_flow_splits(args.variant)
    test = make_pose_test(args.variant)
    geometry = canonical_geometry(args.variant)
    names = geometry.pop("class_names")
    symmetries = geometry.pop("symmetries")
    _save(args.out / "geometry.npz", {**geometry, **{
        f"symmetry_{index}": group for index, group in enumerate(symmetries)
    }})
    for name, split in (("train", train), ("val", validation), ("test", test)):
        _save(args.out / f"{name}.npz", split)
    metadata = {
        "variant": args.variant,
        "class_names": names,
        "quaternion_order": "wxyz",
        "seeds": {"train": 301, "val": 302, "test": 7701},
        "sizes": {name: len(split["class_id"]) for name, split in (
            ("train", train), ("val", validation), ("test", test)
        )},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {args.variant} toy-pose splits to {args.out}")


if __name__ == "__main__":
    main()
