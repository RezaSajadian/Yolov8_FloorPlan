
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


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(mid_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        pool1 = self.maxpool(x)
        pool2 = self.maxpool(pool1)
        pool3 = self.maxpool(pool2)
        
        return self.conv2(torch.cat([x, pool1, pool2, pool3], dim=1))


class YOLOv8n(nn.Module):
    def __init__(self, num_classes: int = 3, input_size: int = 640):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.depth_multiplier = 0.33
        self.width_multiplier = 0.25
        
        base_channels = int(64 * self.width_multiplier)
        
        self.backbone = nn.Sequential(
            ConvBNSiLU(3, base_channels, 3, stride=2, padding=1),
            
            CSPBlock(base_channels, base_channels * 2, num_blocks=1),
            ConvBNSiLU(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
            
            CSPBlock(base_channels * 2, base_channels * 4, num_blocks=2),
            ConvBNSiLU(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
            
            CSPBlock(base_channels * 4, base_channels * 8, num_blocks=2),
            ConvBNSiLU(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),
            
            CSPBlock(base_channels * 8, base_channels * 16, num_blocks=1),
            SPPF(base_channels * 16, base_channels * 16)
        )
        
        self.detect_heads = nn.ModuleList([
            nn.Conv2d(base_channels * 4, 3 * (4 + 1 + num_classes), 1),
            nn.Conv2d(base_channels * 8, 3 * (4 + 1 + num_classes), 1),
            nn.Conv2d(base_channels * 16, 3 * (4 + 1 + num_classes), 1)
        ])
        
        logger.info(f"YOLOv8n model initialized with {num_classes} classes")
        logger.info(f"Input size: {input_size}x{input_size}")
        logger.info(f"Depth multiplier: {self.depth_multiplier}")
        logger.info(f"Width multiplier: {self.width_multiplier}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        current = x
        
        for i, layer in enumerate(self.backbone):
            current = layer(current)
            if i in [4, 6, 8]:  
                features.append(current)
        
        if len(features) < 3:
            while len(features) < 3:
                features.append(features[-1])
        
        p3, p4, p5 = features[-3:]
        
        outputs = [
            self.detect_heads[0](p3),  
            self.detect_heads[1](p4),  
            self.detect_heads[2](p5)
        ]
        
        return outputs
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


def create_yolov8n_model(num_classes: int = 3, input_size: int = 640) -> YOLOv8n:
    if input_size % 32 != 0:
        logger.warning(f"Input size {input_size} is not divisible by 32, may cause issues")
    
    model = YOLOv8n(num_classes=num_classes, input_size=input_size)
    
    logger.info(f"Created YOLOv8n model with {model.get_num_parameters():,} parameters")
    logger.info(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model
