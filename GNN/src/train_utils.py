import torch
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
    path = Path("{}.pth".format(config.run_name)).resolve()
    torch.save(checkpoint, str(path))
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler


def _accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_acc, n_batches = 0., 0., 0
    for imgs, labels in tqdm(loader, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        n_batches += 1
    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_acc, n_batches = 0., 0., 0
    for imgs, labels in tqdm(loader, desc="val"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += _accuracy(logits, labels)
        n_batches += 1
    return total_loss / n_batches, total_acc / n_batches


def train(model, train_loader, val_loader, optimizer, scheduler, run, config):
    model.to(config.device)
    for epoch in range(1, config.n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config.device)
        val_loss, val_acc = val_epoch(model, val_loader, config.device)
        if run is not None:
            run.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        scheduler.step()
        print(
            f"[{epoch}/{config.n_epochs}] "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"acc {train_acc * 100:.1f}%/{val_acc * 100:.1f}%"
        )
