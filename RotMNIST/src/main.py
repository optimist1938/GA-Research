import torch
from pathlib import Path

from src.config import create_argparser
from src.dataset import create_dataloaders
from src.model import RotationNet
from src.train_utils import get_available_device, form_checkpoint, load_checkpoint, train
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run


def main():
    parser = create_argparser()
    config = parser.parse_args()
    config.device = get_available_device()

    train_loader, val_loader = create_dataloaders(config)

    model = RotationNet(
        cnn_channels=config.cnn_channels,
        head=config.head,
        head_hidden=config.head_hidden,
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=config.lr_step_size,
                                                 gamma=config.lr_decay_rate)

    if config.checkpoint:
        model, optimizer, scheduler = load_checkpoint(
            model, optimizer, scheduler, config.checkpoint, config.device)

    run = wandb_create_run(config.run_name)
    pint("wandb")
    wandb_log_code(run, Path("."))

    train(model, train_loader, val_loader, optimizer, scheduler, run, config)

    if config.run_name:
        checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
        wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")

    wandb_finish_run(run)


if __name__ == "__main__":
    main()
