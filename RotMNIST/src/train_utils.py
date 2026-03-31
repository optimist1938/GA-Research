import torch
import numpy as np
import matplotlib.pyplot as plt
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


GRADE_NAMES = ["scalar\n(e0)", "e1", "e2", "bivector\n(e12)"]


@torch.no_grad()
def log_embedding_viz(model, loader, device, run, epoch):
    import wandb
    head = model.head
    if run is None or not hasattr(head, 'embed'):
        return
    # gp1/gp2 for CliffordHead (two-layer block), gp for others
    gp_first = getattr(head, 'gp1', None) or getattr(head, 'gp', None)
    gp_last  = getattr(head, 'gp2', None) or getattr(head, 'gp', None)
    if gp_first is None:
        return

    captured = {}

    def hook_pre(module, inp, out):
        captured['pre_gp'] = inp[0].detach().cpu()

    def hook_post(module, inp, out):
        captured['post_gp'] = out.detach().cpu()

    h1 = gp_first.register_forward_hook(hook_pre)
    h2 = gp_last.register_forward_hook(hook_post)
    model.eval()
    model(next(iter(loader))[0].to(device))
    h1.remove()
    h2.remove()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Epoch {epoch}', fontsize=13)

    pre  = captured['pre_gp'].pow(2).mean(dim=(0, 1)).numpy()
    post = captured['post_gp'].pow(2).mean(dim=(0, 1)).numpy()
    pre  = pre  / pre.sum().clip(1e-8)
    post = post / post.sum().clip(1e-8)

    x = np.arange(4)
    ax = axes[0]
    ax.bar(x - 0.2, pre,  0.4, label='before GP', color='steelblue')
    ax.bar(x + 0.2, post, 0.4, label='after GP',  color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(GRADE_NAMES)
    ax.set_ylabel('fraction of total energy')
    ax.set_title('Grade energy distribution')
    ax.legend()

    W = head.embed.weight.detach().cpu()
    hidden = W.shape[0] // 4
    W = W.reshape(hidden, 4, -1).norm(dim=-1).numpy()
    W = W / W.max().clip(1e-8)

    ax = axes[1]
    im = ax.imshow(W, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["scalar\n(e0)", "e1", "e2", "bivector\n(e12)"])
    ax.set_ylabel('multivector index')
    ax.set_title('Embed weight norm per blade')
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    run.log({"viz/embedding": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


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
        log_embedding_viz(model, val_loader, config.device, run, epoch)
