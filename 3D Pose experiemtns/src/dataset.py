from image2sphere.pascal_dataset import Pascal3D
from torch.utils.data import Dataset

class PascalSanityCheckDataset(Dataset):
    def __init__(self, size=32):
        self.base_dataset = Pascal3D(datasets_dir="/Users/chaykovsky/Downloads", train="train")
        self.size = size


    def __len__(self):
        return self.size


    def __getitem__(self, i):
        if i < self.size:
            return self.base_dataset[i]
        raise ValueError("List Index out of Range")
