import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def form_checkpoint(model, optimizer, scheduler, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    path = Path(f"{config.run_name}.pth").resolve()
    torch.save(checkpoint, str(path))
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return model, optimizer, scheduler


def _angular_error_deg(pred, target):
    theta_pred = torch.atan2(pred[:, 1], pred[:, 0])
    theta_true = torch.atan2(target[:, 1], target[:, 0])
    diff = theta_pred - theta_true
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.abs().mean().item() * 180 / np.pi


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss, total_err, n = 0., 0., 0
    for imgs, labels in tqdm(loader, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_err += _angular_error_deg(pred.detach(), labels)
        n += 1
    return total_loss / n, total_err / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss, total_err, n = 0., 0., 0
    for imgs, labels in tqdm(loader, desc="val"):
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs)
        loss = criterion(pred, labels)
        total_loss += loss.item()
        total_err += _angular_error_deg(pred, labels)
        n += 1
    return total_loss / n, total_err / n


def train(model, train_loader, val_loader, optimizer, scheduler, run, config):
    model.to(config.device)
    for epoch in range(1, config.n_epochs + 1):
        train_loss, train_err = train_epoch(model, train_loader, optimizer, config.device)
        val_loss, val_err = val_epoch(model, val_loader, config.device)
        if run is not None:
            run.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_angle_err_deg": train_err,
                "val_angle_err_deg": val_err,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        scheduler.step()
        print(
            f"[{epoch}/{config.n_epochs}] "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"angle_err {train_err:.1f}°/{val_err:.1f}°"
        )
