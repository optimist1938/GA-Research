import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--head", type=str, default="mlp",
                        choices=["mlp", "clifford", "clifford_scalar"])
    parser.add_argument("--cnn_channels", type=int, default=16)
    parser.add_argument("--head_hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=20)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser
