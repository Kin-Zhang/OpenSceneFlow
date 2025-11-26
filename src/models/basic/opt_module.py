"""
# Created: 2025-11-25 09:16
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * TODO
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: some modules/funcs for optimization and model building.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelGrid(nn.Module):
    """
    Voxel-based scene flow representation.
    Replaces MLP with explicit 3D grid where each vertex stores flow.
    """
    def __init__(self, point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3], 
                 voxel_size=0.5):
        """
        Args:
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: Size of each voxel in meters (default: 0.5m as per paper)
        """
        super().__init__()
        
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # Calculate grid dimensions
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        
        # Number of grid cells along each axis
        self.nx = int((self.x_max - self.x_min) / voxel_size) + 1
        self.ny = int((self.y_max - self.y_min) / voxel_size) + 1
        self.nz = int((self.z_max - self.z_min) / voxel_size) + 1
        
        # Grid vertices store 3D flow vectors (fx, fy, fz)
        # Shape: [nx, ny, nz, 3]
        self.grid = nn.Parameter(
            torch.zeros(self.nx, self.ny, self.nz, 3)
        )
        
        # print(f"Floxels VoxelGrid initialized: "
        #       f"grid shape {self.grid.shape}, "
        #       f"voxel_size {voxel_size}m")
    
    def reset(self):
        """Reset grid to zeros"""
        nn.init.zeros_(self.grid)
    
    def init_weights(self):
        """Initialize grid weights (small random values)"""
        nn.init.normal_(self.grid, mean=0.0, std=0.01)
    
    def _point_to_grid_coords(self, points):
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            points: [N, 3] tensor of (x, y, z) coordinates
        
        Returns:
            grid_coords: [N, 3] tensor of continuous grid indices
        """
        x_grid = (points[:, 0] - self.x_min) / self.voxel_size
        y_grid = (points[:, 1] - self.y_min) / self.voxel_size
        z_grid = (points[:, 2] - self.z_min) / self.voxel_size
        
        return torch.stack([x_grid, y_grid, z_grid], dim=1)
    
    def _trilinear_interpolation(self, grid_coords):
        """
        Perform trilinear interpolation to get flow at arbitrary points.
        
        Args:
            grid_coords: [N, 3] continuous grid coordinates
        
        Returns:
            flow: [N, 3] interpolated flow vectors
        """
        # Get integer parts (lower corner of the voxel)
        x0 = torch.floor(grid_coords[:, 0]).long()
        y0 = torch.floor(grid_coords[:, 1]).long()
        z0 = torch.floor(grid_coords[:, 2]).long()
        
        # Get upper corners
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # Clamp to valid grid indices
        x0 = torch.clamp(x0, 0, self.nx - 1)
        x1 = torch.clamp(x1, 0, self.nx - 1)
        y0 = torch.clamp(y0, 0, self.ny - 1)
        y1 = torch.clamp(y1, 0, self.ny - 1)
        z0 = torch.clamp(z0, 0, self.nz - 1)
        z1 = torch.clamp(z1, 0, self.nz - 1)
        
        # Get fractional parts for interpolation weights
        xd = grid_coords[:, 0] - x0.float()
        yd = grid_coords[:, 1] - y0.float()
        zd = grid_coords[:, 2] - z0.float()
        
        # Get flow values at 8 corners
        c000 = self.grid[x0, y0, z0]  # [N, 3]
        c001 = self.grid[x0, y0, z1]
        c010 = self.grid[x0, y1, z0]
        c011 = self.grid[x0, y1, z1]
        c100 = self.grid[x1, y0, z0]
        c101 = self.grid[x1, y0, z1]
        c110 = self.grid[x1, y1, z0]
        c111 = self.grid[x1, y1, z1]
        
        # Expand dimensions for broadcasting
        xd = xd.unsqueeze(1)  # [N, 1]
        yd = yd.unsqueeze(1)
        zd = zd.unsqueeze(1)
        
        # Trilinear interpolation formula
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        flow = c0 * (1 - zd) + c1 * zd
        
        return flow
    
    def forward(self, points):
        """
        Compute flow for input points via trilinear interpolation.
        
        Args:
            points: [N, 3] tensor of (x, y, z) coordinates
        
        Returns:
            flow: [N, 3] scene flow vectors
        """
        # Convert to grid coordinates
        grid_coords = self._point_to_grid_coords(points)
        
        # Trilinear interpolation
        flow = self._trilinear_interpolation(grid_coords)
        
        return flow

"""
This file is directly copied from: 
https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior
"""    
class Neural_Prior(nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))
    
    def reset(self):
        for layer in self.nn_layers.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def init_weights(self):
        for m in self.nn_layers:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
            
    def forward(self, x):
        """ points -> features
            [N, 3] -> [N, K] -> [N, 3]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x
    
# ANCHOR: early stopping strategy
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
try:
    import FastGeodis # extra package
except:
    print("Warning: FastGeodis package not found. Please install it for DT functionality.")

class DT:
    def __init__(self, pts, pmin, pmax, grid_factor, device='cuda:0'):
        self.device = device
        self.grid_factor = grid_factor
        
        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2
        
        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=self.device)[:-1] / grid_factor + pmin[0]
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=self.device)[:-1] / grid_factor + pmin[1]
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=self.device)[:-1] / grid_factor + pmin[2]
        
        # NOTE: build a binary image first, with 0-value occuppied points
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=device)
        self.pts_sample_idx_x = ((pts[:,0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:,1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:,2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.
        
        iterations = 1
        image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
        self.D = FastGeodis.generalised_geodesic3d(
            image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, 0.0, iterations
        ).squeeze()
            
    def torch_bilinear_distance(self, Y, truncate_dist=-1):
        H, W, D = self.D.size()
        target = self.D[None, None, ...]
        
        sample_x = ((Y[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((Y[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((Y[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)
        
        sample = torch.cat([sample_x, sample_y, sample_z], -1)
        
        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1
        
        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)
        
        # NOTE: reshape to match 5D volumetric input
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)
        if truncate_dist > 0:
            dist = torch.clamp(dist, max=truncate_dist)
        return dist