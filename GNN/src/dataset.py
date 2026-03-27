import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_cifar10_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(config.data_dir, train=True,
                                     transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(config.data_dir, train=False,
                                   transform=val_transform, download=True)

    if config.sanity_check:
        train_dataset = Subset(train_dataset, range(config.batch_size * 4))
        val_dataset = Subset(val_dataset, range(config.batch_size * 2))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader
