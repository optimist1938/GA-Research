import argparse
from dataclasses import dataclass

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None, required=False)
    parser.add_argument("--sanity_check", action="store_true")
    return parser


@dataclass
class JsonYamlevich:
    n_epochs : int = 1
    batch_size : int = 32
    path_to_datasets = "/Users/chaykovsky/Downloads/"
