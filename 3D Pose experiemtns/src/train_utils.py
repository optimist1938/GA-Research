import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion):
    total_loss = 0
    n_objects = 0

    model.train()
    for data in tqdm(loader):
        img, targets = data["img"], data["rot"]
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss
        n_objects += len(img)

    total_loss /= n_objects

    return total_loss


def train(model, loader, optimizer, criterion, run):
    pass
