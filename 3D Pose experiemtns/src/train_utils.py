import torch
from tqdm import tqdm
from pathlib import Path

def form_checkpoint(model, optimizer, scheduler, config):
    checkpoint = {
        "model" : model.state_dict(),
        "scheduler" : scheduler.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    path = Path("{}.pth".format(config.run_name)).resolve()
    torch.save(checkpoint, path.__str__())
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler



def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def get_currently_used_device(model):
    return next(model.parameters()).device


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, config):
    total_loss = 0
    n_objects = 0

    model.train()
    for data in tqdm(loader):
        img, targets = data["img"].to(config.device), data["rot"].to(config.device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss
        n_objects += len(img)

    total_loss /= n_objects

    return total_loss


@torch.no_grad()
def validate_epoch(model, loader, criterion, config):
    total_loss = 0
    n_objects = 0

    model.eval()
    for data in tqdm(loader):
        img, targets = data["img"].to(config.device), data["rot"].to(config.device)
        outputs = model(img)
        loss = criterion(outputs, targets)

        total_loss += loss
        n_objects += len(img)

    total_loss /= n_objects

    return total_loss


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, config):
    model.to(config.device)
    for i in range(config.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        val_loss = validate_epoch(model, val_loader, criterion, config)
        if run is not None:
            run.log({
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "learning_rate" : scheduler.get_last_lr(),
                "gradient_norm" : grad_norm(model)
            })
        scheduler.step()
        print("Training on {} epoch {} / {}. Train loss {}, val loss as low as {}".format(config.device, i + 1, config.n_epochs, train_loss, val_loss))
