import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CliffordCifar10
from model import CIFAR10_model
from train_utils import train
from config import create_argparser, JsonYamlevich
from pose_3d.src.wandb_utils import wandb_create_run, wandb_finish_run

from algebra.cliffordalgebra import CliffordAlgebra

def main():
    args = create_argparser().parse_args()
    config = JsonYamlevich(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        path_to_datasets=args.path_to_datasets,
        run_name=args.run_name,
        hidden_channels=args.hidden_channels
    )

    run = wandb_create_run(config.run_name)

    ca = CliffordAlgebra((1, 1))
    
    train_set = CliffordCifar10(root=config.path_to_datasets, train=True, rotate=False)
    test_init_set = CliffordCifar10(root=config.path_to_datasets, train=False, rotate=False)
    test_rot_set = CliffordCifar10(root=config.path_to_datasets, train=False, rotate=True)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_init_loader = DataLoader(test_init_set, batch_size=config.batch_size, shuffle=False)
    test_rot_loader = DataLoader(test_rot_set, batch_size=config.batch_size, shuffle=False)

    model = CIFAR10_model(
        ca=ca, 
        in_channels=5, 
        hidden_channels=config.hidden_channels, 
        out_classes=config.out_classes
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        train_loader=train_loader,
        test_init_loader=test_init_loader,
        test_rot_loader=test_rot_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        run=run,
        config=config
    )

    wandb_finish_run(run)

if __name__ == "__main__":
    main()