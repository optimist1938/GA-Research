import torch
from pathlib import Path

from image2sphere.predictor import I2S

from src.config import create_argparser
from src.dataset import create_dataloaders, create_symsol_dataloaders, create_modelnet10_dataloaders
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run
from src.train_utils import form_checkpoint, get_available_device
from src.main_cliffordnet import train_epoch, val_epoch, train, create_argparser_i2s


def create_argparser_baseline():
    parser = create_argparser_i2s()
    parser.add_argument("--encoder", type=str, default="resnet50")
    return parser


def main():
    parser = create_argparser_baseline()
    config = parser.parse_args()
    config.device = get_available_device()

    if config.dataset == "modelnet10":
        train_loader, val_loader = create_modelnet10_dataloaders(config)
    elif config.dataset == "symsol":
        train_loader, val_loader = create_symsol_dataloaders(config)
    else:
        train_loader, val_loader = create_dataloaders(config)

    model = I2S(
        sphere_fdim=config.sphere_fdim,
        encoder=config.encoder,
        lmax=config.lmax,
        f_hidden=config.f_hidden,
        train_grid_rec_level=config.train_grid_rec_level,
        eval_grid_rec_level=config.eval_grid_rec_level,
    )
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.lr_step_size,
                                                gamma=config.lr_decay_rate)
    run = wandb_create_run(config.run_name)
    wandb_log_code(run, Path("."))

    train(model, train_loader, val_loader, optimizer, scheduler, run, config)

    checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
    wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
