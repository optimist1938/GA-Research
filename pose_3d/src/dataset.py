from image2sphere.pascal_dataset import Pascal3D
from image2sphere.dataset import SymsolDataset, ModelNet10Dataset
from torch.utils.data import Dataset, DataLoader

class PascalSanityCheckDataset(Dataset):
    def __init__(self, config):
        self.base_dataset = Pascal3D(datasets_dir=config.path_to_datasets, train="train")
        self.size = config.batch_size


    def __len__(self):
        return self.size


    def __getitem__(self, i):
        if i < self.size:
            return self.base_dataset[i]
        raise ValueError("List Index out of Range")


def create_dataloaders(config):
    if not config.sanity_check:
        train_dataset = Pascal3D(config.path_to_datasets, True)
        val_dataset = Pascal3D(config.path_to_datasets, False)
    else:
        train_dataset = val_dataset = PascalSanityCheckDataset(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader


def create_modelnet10_dataloaders(config):
    train_dataset = ModelNet10Dataset(config.path_to_datasets, train=True,
                                      limited=config.modelnet_limited)
    val_dataset = ModelNet10Dataset(config.path_to_datasets, train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader


def create_symsol_dataloaders(config):
    train_dataset = SymsolDataset(config.path_to_datasets, train=True,
                                  set_number=config.symsol_set, num_views=config.symsol_num_views)
    val_dataset = SymsolDataset(config.path_to_datasets, train=False,
                                set_number=config.symsol_set, num_views=5000)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    return train_loader, val_loader
