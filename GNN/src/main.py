import torch
from pathlib import Path

from src.config import create_argparser
from src.dataset import create_cifar10_dataloaders
from src.model import VigCGENNClassifier
from src.train_utils import get_available_device, form_checkpoint, load_checkpoint, train
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run


def main():
    parser = create_argparser()
    config = parser.parse_args()
    config.device = get_available_device()

    train_loader, val_loader = create_cifar10_dataloaders(config)

    model = VigCGENNClassifier(
        embed_dim=config.embed_dim,
        n_layers=config.n_layers,
        k=config.k,
        hidden_dim=config.hidden_dim,
        patch_size=config.patch_size,
        img_size=config.img_size,
        drop_path_rate=config.drop_path_rate,
    )

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
