"""
Training package for YOLOv8n

Contains:
- PyTorch Lightning modules
- Training scripts
- Training utilities
"""

from .lightning_module import YOLOv8nLightningModule, create_lightning_module

__all__ = [
    'YOLOv8nLightningModule',
    'create_lightning_module'
]
