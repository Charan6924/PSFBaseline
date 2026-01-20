import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelEstimator(nn.Module):
    def __init__(self, img_size=512, num_control_points=15, dropout_rate=0.3):
        super().__init__()
        self.img_size = img_size
        self.num_control_points = num_control_points
        
        self.conv1 = self._conv_block(1, 32, dropout_rate * 0.3)
        self.conv2 = self._conv_block(32, 64, dropout_rate * 0.5)
        self.conv3 = self._conv_block(64, 128, dropout_rate * 0.7)
        self.conv4 = self._conv_block(128, 256, dropout_rate * 0.8)
        self.conv5 = self._conv_block(256, 256, dropout_rate)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_control_points)
        )
        
        self.activation = nn.Softplus(beta=2.0)
        self._create_radial_grid(img_size)
    
    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout)
        )
    
    def _create_radial_grid(self, img_size):
        y, x = torch.meshgrid(
            torch.arange(img_size), 
            torch.arange(img_size), 
            indexing='ij'
        )
        center = img_size // 2
        r = torch.sqrt((x - center)**2 + (y - center)**2)
        r_norm = (r / (img_size / 2)).clamp(0, 1) * 2 - 1
        grid = torch.stack([r_norm, torch.zeros_like(r_norm)], dim=-1)
        self.register_buffer('grid', grid.unsqueeze(0))
    
    def build_2d_mtf(self, control_points):
        B = control_points.shape[0]
        profile = control_points.view(B, 1, 1, -1)
        batch_grid = self.grid.expand(B, -1, -1, -1) # type: ignore
        return F.grid_sample(profile, batch_grid, mode='bilinear',
                           align_corners=True, padding_mode='border')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        gap = self.gap(x).flatten(1)
        gmp = self.gmp(x).flatten(1)
        x = torch.cat([gap, gmp], dim=1)
        
        control = self.fc_layers(x)
        control = self.activation(control)
        control = control / (control.max(dim=1, keepdim=True).values + 1e-8)
        control = torch.cat([
            torch.ones(control.shape[0], 1, device=control.device),
            control[:, 1:]
        ], dim=1)
        
        mtf_2d = self.build_2d_mtf(control)
        return mtf_2d