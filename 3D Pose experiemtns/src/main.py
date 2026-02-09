import torch
import torch.nn as nn
from pathlib import Path

from clifford.algebra.cliffordalgebra import CliffordAlgebra

from src.config import create_argparser
from src.dataset import create_dataloaders
from src.model import TralaleroCompetitor, MLPBaseline, I2S
from src.train_utils import train, form_checkpoint, get_available_device,load_checkpoint
from src.evaluation_metrics import calculate_evaluation_metrics,create_technical_matrices
from src.wandb_utils import (
    wandb_create_run,
    wandb_log_code,
    wandb_log_artifact,
    wandb_finish_run,
)


def _make_algebra():
    return CliffordAlgebra((1, 1, 1, 1, -1))


def instantiate(config):
    train_loader, val_loader = create_dataloaders(config)
    print("Created Tralaloaders")

    algebra = _make_algebra()

    if config.model == "tralalero":
        model = TralaleroCompetitor(algebra)
    elif config.model == "mlp":
        model = MLPBaseline()
    elif config.model == "i2s":
        model = I2S(
            algebra=algebra,
            lmax=config.lmax,
            rec_level=config.rec_level,
            n_mv=config.n_mv,
            hidden_dim=config.hidden_dim,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")
    
    config.device = get_available_device()
    
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)

    if config.loss == "mse":
        criterion = nn.MSELoss()
    elif config.loss == "prob":
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        raise ValueError(f"Unknown loss: {config.loss}")

    run = wandb_create_run(config.run_name)
    print("W&B logging set up completed")

    return train_loader, val_loader, model, optimizer, scheduler, criterion, run


def main():
    parser = create_argparser()
    config = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, model, optimizer, scheduler, criterion, run = instantiate(config)
    create_technical_matrices(config)

    path = config.path_to_checkpoint
    if path is not None:
        load_checkpoint(model, optimizer, scheduler, path , config.device)
        print("Checkpoint successfully loaded. Starting evaluation")
        torch.save(torch.tensor(calculate_evaluation_metrics(model,val_loader,config)),"res.pth")

    wandb_log_code(run, Path("."))
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config)

    checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
    wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
