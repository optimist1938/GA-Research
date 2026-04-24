import torch
import torch.nn as nn
from typing import Optional
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np 
import open3d as o3d


from clifford.models.modules.linear import MVLinear
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.mvlayernorm import MVLayerNorm
from clifford.models.modules.mvsilu import MVSiLU
from clifford.algebra.cliffordalgebra import CliffordAlgebra


from src.model import TralaleroTralala



class PointCloudProcessor:
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
        self.model = self.pipe.model
        self.image_processor = self.pipe.image_processor
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x : torch.tensor):
        '''
        Arguments:
            x : torch.Tensor of size (B, C, H, W)
        '''
        return self.model(x).predicted_depth.detach()

    def visualize_matplotlib(self, pcd: o3d.geometry.PointCloud, sample_size: int = 10000):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Depth')
        ax.set_title('3D Point Cloud')



class I2P(nn.Module):
    def __init__(self, device : torch.device = torch.device("mps"), batch_size : Optional[int] = None):
        super().__init__()
        self.n_pixels = 224
        self.depth_anything_model = PointCloudProcessor(device=device, model_size="small")
        self.device = device
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2)
        )
        self.projection = MVLinear(self.algebra, in_features=55 * 55, out_features=512)
        self.tralalero = TralaleroTralala(self.algebra, in_features=512, out_features=1, hidden_dim=[256, 128, 64])
        self.head = nn.Linear(8, 9)
        self.batch_size = batch_size
        self.batched_point_clouds = None
        self._create_batched_clouds()

    def _create_batched_clouds(self):
        if self.batch_size is None:
            return
        v_coords, u_coords = np.indices((55, 55))
        v_coords, u_coords = torch.tensor(v_coords).unsqueeze(2).to(self.device), torch.tensor(u_coords).unsqueeze(2).to(self.device)
        depth_map = torch.zeros((55, 55, 1)).to(self.device)
        point_cloud = torch.cat((v_coords, u_coords, depth_map), dim=-1)
        self.batched_point_clouds = torch.cat([point_cloud.unsqueeze(0) for _ in range(self.batch_size)], dim=0)


    def forward(self, x, cls_info : Optional[torch.Tensor] = None, timeit=False):
        '''
        Args:
            x : torch.Tensor of size (B, C, H, W)
        '''
        import time
        import pandas as pd
        
        timings = []
        batch_size = x.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self._create_batched_clouds()

        t0 = time.perf_counter()
        x = self.depth_anything_model(x)
        timings.append(('x = self.depth_anything_model(x)', time.perf_counter() - t0))

        x = self.adapter(x.unsqueeze(1))
        self.batched_point_clouds[:, :, :, 2] = x.squeeze(1)
        x = self.algebra.embed_grade(self.batched_point_clouds.reshape(batch_size, -1, 3), 1)

        t0 = time.perf_counter()
        x = self.projection(x)
        timings.append(('x = self.projection(x)', time.perf_counter() - t0))

        t0 = time.perf_counter()
        x = self.tralalero(x)
        timings.append(('x = self.tralalero(x)', time.perf_counter() - t0))

        x = x.squeeze(1)
        x = self.head(x)
        out = x.reshape(batch_size, 3, 3)
        if not timeit:
            return out
        return out, pd.DataFrame(timings, columns=['line', 'time_sec'])
