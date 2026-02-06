import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision import datasets
import torchvision.transforms.functional as TF

class CliffordCifar10(Dataset):
    def __init__(self, root, train=True, rotate=False):
        self.post_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train = train
        self.rotate = rotate
        self.data = datasets.CIFAR10(root, train=train, download=True)
        coords = torch.linspace(-1, 1, 32)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij") 
        self.grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        if self.rotate:
            img = TF.rotate(img, angle=45)
            
        img_tensor = self.post_transforms(img)
        img_flat = img_tensor.reshape(3, -1).permute(1, 0) # [1024, 3]
        
        x_combined = torch.cat([img_flat, self.grid], dim=-1) # [1024, 5]
        
        return x_combined, label