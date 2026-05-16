import argparse
from dataclasses import dataclass


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--path_to_datasets", type=str, required=True)
    parser.add_argument("--path_to_checkpoint",type=str,default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="3D Pose Estimation")
    parser.add_argument("--wandb_entity", type=str, default="clifforders")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--platform",type=str,choices=["kaggle","colab"],default="kaggle")

    parser.add_argument("--model", type=str, default="tralalero",
                        choices=["tralalero", "mlp", "vit_baseline", "i2s", "ga_i2s", "i2s_resnet"])
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "geodesic", "prob", "rotor", "mv_rotor"])
    parser.add_argument("--encoder", type=str, default="resnet",
                        choices=["resnet", "ga", "ga_canonical"])
    parser.add_argument("--algebra_dim", type=int, default=3)

    # ViT baseline
    parser.add_argument(
        "--vit_backbone_type",
        type=str,
        default="vit",
        choices=["vit", "depth_anything_v2"],
    )
    parser.add_argument("--vit_model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--vit_layers", type=int, nargs="+", default=[-1, -3, -6, -9])
    parser.add_argument("--freeze_vit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vit_pooling_type",
        type=str,
        default="mean",
        choices=["mean", "attention", "transformer_attention", "convolution"],
    )
    parser.add_argument("--vit_num_transformer_layers", type=int, default=1)
    parser.add_argument("--vit_transformer_nhead", type=int, default=8)
    parser.add_argument("--vit_transformer_ff_dim", type=int, default=1024)
    parser.add_argument("--vit_transformer_dropout", type=float, default=0.1)

    # I2S
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--rec_level", type=int, default=3)
    parser.add_argument("--n_mv", type=int, default=8)
    parser.add_argument("--ga_pool_hw", type=int, nargs=2, default=[28, 28])
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[32])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--i2s_resnet_output_mode",
        type=str,
        default="auto",
        choices=["auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"],
    )
    parser.add_argument(
        "--i2s_resnet_backbone_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "convnext_tiny"],
    )
    parser.add_argument(
        "--i2s_resnet_pretrained_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--i2s_resnet_freeze_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--i2s_resnet_use_positional_encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--i2s_resnet_mv_per_position",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--i2s_resnet_adapter_mid_channels",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--i2s_resnet_adapter_high_channels",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--i2s_resnet_adapter_output_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--i2s_resnet_ga_head_type",
        type=str,
        default="tralalero",
        choices=["tralalero", "transformer_like", "reduced", "residual_gp"],
    )
    parser.add_argument(
        "--i2s_resnet_ga_head_mixing_layer",
        type=str,
        default="gp",
        choices=["gp", "mvlinear", "linear"],
    )
    parser.add_argument(
        "--i2s_resnet_ga_num_blocks",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--i2s_resnet_ga_head_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--i2s_resnet_ga_head_use_layer_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--ram_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multi_gpu", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr", type=float, default=1e-3)

    return parser


@dataclass
class JsonYamlevich:
    n_epochs: int = 1
    batch_size: int = 32
    path_to_datasets: str = "/Users/chaykovsky/Downloads/"
    wandb_project: str = "CLIP"
    wandb_entity: str | None = "clifforders"
    wandb_group: str | None = None
    vit_backbone_type: str = "vit"
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    vit_layers: tuple = (-1, -3, -6, -9)
    freeze_vit: bool = True
    vit_pooling_type: str = "mean"
    vit_num_transformer_layers: int = 1
    vit_transformer_nhead: int = 8
    vit_transformer_ff_dim: int = 1024
    vit_transformer_dropout: float = 0.1
    i2s_resnet_ga_num_blocks: int = 2
    i2s_resnet_ga_head_dropout: float = 0.0
    i2s_resnet_ga_head_use_layer_norm: bool = False
