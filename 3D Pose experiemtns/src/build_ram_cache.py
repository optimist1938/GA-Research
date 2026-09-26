"""Build the --ram_memory Pascal3D tensors once and write them to --ram_cache_save_dir.

    python -m src.build_ram_cache --path_to_datasets=... --ram_memory --ram_cache_save_dir=/kaggle/working/ram_cache

Upload the resulting folder as a Kaggle dataset and pass it to training as --ram_cache_dir.
CPU only, so it does not use GPU quota.
"""
import time

from src.config import create_argparser
from src.dataset import _ram_dataset


def main():
    config = create_argparser().parse_args()
    assert config.ram_cache_save_dir, "pass --ram_cache_save_dir"
    config.ram_cache_dir = None   # always rebuild from the raw dataset
    t0 = time.time()
    for split in ("train", "val"):
        _ram_dataset(split, config, use_multiprocessing=config.multiprocessing if split == "train" else False)
    print(f"[timing] cache built and saved in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
