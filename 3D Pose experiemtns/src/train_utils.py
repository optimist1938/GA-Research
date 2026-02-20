import torch
from tqdm import tqdm
from pathlib import Path
from src.evaluation_metrics import calculate_evaluation_metrics
import numpy as np


def form_checkpoint(model, optimizer, scheduler, config):
    checkpoint = {
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(config),
    }
    path = Path(f"{config.run_name}.pth").resolve()
    torch.save(checkpoint, path.__str__())
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        scheduler.load_state_dict(checkpoint["scheduler"])
    except:
        scheduler = scheduler
    return model, optimizer, scheduler


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_currently_used_device(model):
    return next(model.parameters()).device


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_loss(model, data, criterion, config):
    img = data["img"].to(config.device)
    targets = data["rot"].to(config.device)
    outputs = model(img)
    if config.loss == "prob":
        idx = model.get_nearest_idx(targets)
        return criterion(outputs, idx)
    return criterion(outputs,targets)


def train_epoch(model, loader, optimizer, criterion, config):
    total_loss = 0.0
    n_objects = 0
    device_type = "cuda" if config.device.type == "cuda" else "cpu"
    
    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None
    
    model.train()
    for data in tqdm(loader):
        optimizer.zero_grad(set_to_none=True)

        
        loss = _compute_loss(model, data, criterion, config)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    return total_loss / max(n_objects, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, config):
    total_loss = 0.0
    n_objects = 0

    model.eval()
    for data in tqdm(loader):
        loss = _compute_loss(model, data, criterion, config)

        total_loss += loss
        n_objects += len(data["img"])

    total_loss /= n_objects

    return total_loss


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config):

    for i in range(config.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        val_loss = validate_epoch(model, val_loader, criterion, config)
        mre =  np.median(calculate_evaluation_metrics(model, val_loader, config)).__float__()

        if run is not None:
            run.log({
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "median_rotation_error" : mre,
                "learning_rate" : scheduler.get_last_lr()[0],
                "gradient_norm" : grad_norm(model)
            })
        scheduler.step()
        print(
            f"Training on {config.device} epoch {i + 1} / {config.n_epochs}. "
            f"Train loss {train_loss}, val loss {val_loss}"
        )
