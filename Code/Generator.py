import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline
import torch.nn.functional as F

class KernelEstimator(nn.Module):
  def __init__(self,img_size=512, num_control_points=15):
    super().__init__()
    self.img_size = img_size
    self.num_control_points = num_control_points
    self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, stride = 2),    #512 -> 256
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        
        nn.Conv2d(32, 64, 3, padding=1, stride=2),   #256 -> 128
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        
        nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #128 -> 64
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        
        nn.Conv2d(128, 64, 3, padding=1, stride = 2),  #64 -> 32
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),

        nn.Conv2d(64, 32, 3, padding=1, stride=2),   #32 -> 16
        nn.BatchNorm2d(32),
        nn.LeakyReLU()
    )

    self.pool = nn.AdaptiveAvgPool2d(1)
    self.flatten = nn.Flatten() # [1,32,363] -> [32*363]
    self.fc_head = nn.Linear(32,self.num_control_points)  # 15 points
    self.activation = nn.Softplus(beta = 2.0)
    y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
    center = img_size // 2
    r = torch.sqrt((x - center)**2 + (y - center)**2)
    r_norm = (r / (img_size / 2)) * 2 - 1 
    self.grid = torch.stack([r_norm, torch.zeros_like(r_norm)], dim=-1)
    self.register_buffer('grid', self.grid.unsqueeze(0))

  def build_2d_mtf(self, control_points):
        B = control_points.shape[0]
        profile = control_points.view(B, 1, 1, -1)
        batch_grid = self.grid.expand(B, -1, -1, -1)
        return F.grid_sample(profile, batch_grid, align_corners=True, padding_mode='border')

  def forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    x = self.flatten(x)
    control = self.fc_head(x)
    control = self.activation(control)
    control = control / (control.max(dim=1, keepdim=True).values + 1e-8)
    mtf_2d = self.build_2d_mtf(control)
        
    return mtf_2d