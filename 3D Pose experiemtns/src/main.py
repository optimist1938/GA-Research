import torch
import torch.nn as nn
from src.config import create_argparser
from src.dataset import create_dataloaders
from src.model import TralaleroCompetitor, MLPBaseline
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run
from src.train_utils import train, form_checkpoint, get_available_device
from pathlib import Path
from clifford.algebra.cliffordalgebra import CliffordAlgebra

def instantiate_tralalero(config):
    train_loader, val_loader = create_dataloaders(config)
    print("Created two pretty Tralaloaders!")
    algebra = CliffordAlgebra((1, 1, 1, 1, -1))
    # tralalero = TralaleroCompetitor(algebra)
    tralalero = MLPBaseline()
    print("Spawned Tralelero")
    optimizer = torch.optim.Adam(tralalero.parameters())
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    criterion = nn.MSELoss()
    run = wandb_create_run(config.run_name)
    print("W&B logging set up completed")
    return train_loader, val_loader, tralalero, optimizer, scheduler, criterion, run


def main():
    parser = create_argparser()
    config = parser.parse_args()
    config.device = get_available_device()
    train_loader, val_loader, model, optimizer, scheduler, criterion, run = instantiate_tralalero(config)
    wandb_log_code(run, Path("."))
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config)
    checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
    wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
