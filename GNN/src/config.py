import argparse


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download/load CIFAR-10")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbours in KNN graph")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="CGENN hidden multivector dimension")
    parser.add_argument("--img_size", type=int, default=32,
                        help="Input image size (CIFAR-10 default: 32)")
    parser.add_argument("--patch_size", type=int, default=4,
                        help="Patch size; img_size must be divisible by patch_size")
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step_size", type=int, default=50)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser
