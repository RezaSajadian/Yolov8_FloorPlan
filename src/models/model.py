
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.main_branch = nn.Sequential(
            ConvBNSiLU(in_channels, mid_channels, 1),
            *[ConvBNSiLU(mid_channels, mid_channels, 3, padding=1) for _ in range(num_blocks)]
        )
        
        self.shortcut = ConvBNSiLU(in_channels, mid_channels, 1)
        
        self.final_conv = ConvBNSiLU(mid_channels * 2, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into two branches
        main_out = self.main_branch(x)
        shortcut_out = self.shortcut(x)
        
        # Concatenate along channel dimension
        # adds a new dimension to combine features from both branches
        combined = torch.cat([main_out, shortcut_out], dim=1)
        return self.final_conv(combined)
