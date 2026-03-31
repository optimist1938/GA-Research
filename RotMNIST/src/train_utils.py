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


GRADE_NAMES = ["scalar (e0)", "e1", "e2", "bivector (e12)"]


@torch.no_grad()
def log_grade_energy(model, loader, device, run, epoch):
    head = model.head
    captured = {}
    def hook_pre_gp(module, inp, out):
        captured['pre_gp'] = inp[0].detach().cpu()

    def hook_post_gp(module, inp, out):
        captured['post_gp'] = out.detach().cpu()

    h1 = head.gp.register_forward_hook(hook_pre_gp)
    h2 = head.gp.register_forward_hook(hook_post_gp)

    imgs, _ = next(iter(loader))
    model.eval()
    model(imgs.to(device))

    h1.remove()
    h2.remove()

    logs = {}
    for tag, mv in captured.items():
        energy = mv.pow(2).mean(dim=(0, 1))  
        energy = energy / energy.sum().clamp(min=1e-8) 
        for k, name in enumerate(GRADE_NAMES):
            logs[f"grade_energy/{tag}/{name}"] = energy[k].item()

    if run is not None:
        run.log(logs, step=epoch)

    pre  = [captured['pre_gp'].pow(2).mean(dim=(0,1))[k].item()  for k in range(4)]
    post = [captured['post_gp'].pow(2).mean(dim=(0,1))[k].item() for k in range(4)]
    print(f"  grade energy  {'':10s}  " + "  ".join(f"{n:>16s}" for n in GRADE_NAMES))
    print(f"  before GP     {'':10s}  " + "  ".join(f"{v:>16.4f}" for v in pre))
    print(f"  after  GP     {'':10s}  " + "  ".join(f"{v:>16.4f}" for v in post))


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
        log_grade_energy(model, val_loader, config.device, run, epoch)
