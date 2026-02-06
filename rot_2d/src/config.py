import argparse
import torch
from dataclasses import dataclass

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="clifford_cifar10")
    parser.add_argument("--hidden_channels", type=int, default=1024)
    return parser

@dataclass
class JsonYamlevich:
    n_epochs: int
    batch_size: int
    lr: float
    path_to_datasets: str
    run_name: str
    hidden_channels: int
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_classes: int = 10