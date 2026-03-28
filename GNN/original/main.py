import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path


def _import_vig(vig_repo):
    vig_path = str(Path(vig_repo) / "vig_pytorch")
    if vig_path not in sys.path:
        sys.path.insert(0, vig_path)
    from vig import vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu
    return vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu

from src.dataset import create_cifar10_dataloaders
from src.train_utils import get_available_device, form_checkpoint, load_checkpoint, train
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run


class VigCifar10(nn.Module):
    def __init__(self, model_size = "ti", vig_repo = "./Efficient-AI-Backbones"):
        super().__init__()
        pvig_ti, pvig_s, pvig_b = _import_vig(vig_repo)
        vig_models = {"ti": pvig_ti, "s": pvig_s, "b": pvig_b}
        self.backbone = vig_models[model_size]()
        self._adapt_stem_for_small_images()
        last_idx = max(
            i for i, m in enumerate(self.backbone.prediction) if isinstance(m, nn.Conv2d)
        )
        in_ch = self.backbone.prediction[last_idx].in_channels
        self.backbone.prediction[last_idx] = nn.Conv2d(in_ch, 10, kernel_size=1, bias=True)

    def _adapt_stem_for_small_images(self):
        stem = self.backbone.stem
        convs = stem.convs if hasattr(stem, 'convs') else stem
        changed = 0
        for m in convs.modules():
            if isinstance(m, nn.Conv2d) and m.stride[0] == 2 and changed == 0:
                m.stride = (1, 1)
                m.padding = (m.kernel_size[0] // 2, m.kernel_size[0] // 2)
                changed += 1

    def forward(self, x):
        return self.backbone(x)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--model_size", type=str, default="ti",
                        choices=["ti", "s", "b"],
                        help="ViG variant: ti=tiny, s=small, b=base")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step_size", type=int, default=50)
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--vig_repo", type=str,
                        default="./Efficient-AI-Backbones",
                        help="Path to cloned huawei-noah/Efficient-AI-Backbones repo")
    return parser


def main():
    parser = create_argparser()
    config = parser.parse_args()
    config.device = get_available_device()

    train_loader, val_loader = create_cifar10_dataloaders(config)

    model = VigCifar10(model_size=config.model_size, vig_repo=config.vig_repo)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=config.lr_step_size,
                                                 gamma=config.lr_decay_rate)

    if config.checkpoint:
        model, optimizer, scheduler = load_checkpoint(
            model, optimizer, scheduler, config.checkpoint, config.device)

    run = wandb_create_run(config.run_name)
    wandb_log_code(run, Path("."))

    train(model, train_loader, val_loader, optimizer, scheduler, run, config)

    if config.run_name:
        checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
        wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")

    wandb_finish_run(run)


if __name__ == "__main__":
    main()
