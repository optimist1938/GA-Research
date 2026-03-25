import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.config import create_argparser
from src.dataset import create_dataloaders
from src.cliffordnet_model import CliffordNetI2S
from src.wandb_utils import wandb_create_run, wandb_log_code, wandb_log_artifact, wandb_finish_run
from src.train_utils import form_checkpoint, get_available_device


def create_argparser_i2s():
    parser = create_argparser()
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--shifts_n", type=int, default=2,
                        help="shifts=[1,2,...,2^(shifts_n-1)]")
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--sphere_fdim", type=int, default=512)
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--f_hidden", type=int, default=8)
    parser.add_argument("--train_grid_rec_level", type=int, default=3)
    parser.add_argument("--eval_grid_rec_level", type=int, default=5)
    return parser


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, all_acc, n_batches = 0., [], 0
    for data in tqdm(loader, desc="train"):
        img = data["img"].to(device)
        cls = data["cls"].to(device)
        rot = data["rot"].to(device)
        optimizer.zero_grad()
        loss, acc = model.compute_loss(img, cls, rot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_acc.append(acc)
        n_batches += 1
    return total_loss / n_batches, np.degrees(np.median(np.concatenate(all_acc)))


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss, all_acc, n_batches = 0., [], 0
    for data in tqdm(loader, desc="val"):
        img = data["img"].to(device)
        cls = data["cls"].to(device)
        rot = data["rot"].to(device)
        loss, acc = model.compute_loss(img, cls, rot)
        total_loss += loss.item()
        all_acc.append(acc)
        n_batches += 1
    return total_loss / n_batches, np.degrees(np.median(np.concatenate(all_acc)))


def train(model, train_loader, val_loader, optimizer, scheduler, run, config):
    model.to(config.device)
    for epoch in range(1, config.n_epochs + 1):
        train_loss, train_err_deg = train_epoch(model, train_loader, optimizer, config.device)
        val_loss, val_err_deg = val_epoch(model, val_loader, config.device)
        if run is not None:
            run.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rot_err_deg": train_err_deg,
                "val_rot_err_deg": val_err_deg,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        scheduler.step()
        print(
            f"[{epoch}/{config.n_epochs}] "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"rot_err {train_err_deg:.1f}°/{val_err_deg:.1f}°"
        )


def main():
    parser = create_argparser_i2s()
    config = parser.parse_args()
    config.device = get_available_device()

    train_loader, val_loader = create_dataloaders(config)
    shifts = [1 << i for i in range(config.shifts_n)] # this is used to specify how many bivectors out of binom(embed_dim, 2) we are goiong to compute
    model = CliffordNetI2S(
        embed_dim=config.embed_dim,
        depth=config.depth,
        shifts=shifts,
        drop_path_rate=config.drop_path_rate,
        img_size=config.img_size,
        patch_size=config.patch_size,
        sphere_fdim=config.sphere_fdim,
        lmax=config.lmax,
        f_hidden=config.f_hidden,
        train_grid_rec_level=config.train_grid_rec_level,
        eval_grid_rec_level=config.eval_grid_rec_level,
    )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    run = wandb_create_run(config.run_name)
    wandb_log_code(run, Path("."))

    train(model, train_loader, val_loader, optimizer, scheduler, run, config)

    checkpoint_path = form_checkpoint(model, optimizer, scheduler, config)
    wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
