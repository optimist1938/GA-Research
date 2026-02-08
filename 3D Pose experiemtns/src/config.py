import argparse
from dataclasses import dataclass


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")

    parser.add_argument("--model", type=str, default="tralalero",
                        choices=["tralalero", "mlp", "i2s"])
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "prob"])

    # I2S
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--rec_level", type=int, default=3)
    parser.add_argument("--n_mv", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--ram_memory",type=bool,default=False)

    parser.add_argument("--lr", type=float, default=1e-3)

    return parser


@dataclass
class JsonYamlevich:
    n_epochs: int = 1
    batch_size: int = 32
    path_to_datasets: str = "/Users/chaykovsky/Downloads/"
