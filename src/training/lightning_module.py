import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

from ..models.yolo_model import YOLOv8n
from ..models.working_loss import WorkingYOLOLoss
from ..utils.logger import get_logger
from ..utils.config import ConfigManager

logger = get_logger(__name__)


class YOLOv8nLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int = 3, input_size: int = 640, learning_rate: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Model and loss function
        self.model = YOLOv8n(num_classes=num_classes, input_size=input_size)
        self.loss_fn = WorkingYOLOLoss(num_classes=num_classes)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []

        self.example_input_array = torch.randn(1, 3, input_size, input_size)
        
        logger.info(f"Lightning module initialized with {num_classes} classes")
        logger.info(f"Input size: {input_size}x{input_size}")
        logger.info(f"Learning rate: {learning_rate}")
    
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:

        images, labels = batch
        
        # Forward pass
        predictions = self(images)
        
        # Calculate loss
        loss_dict = self.loss_fn(predictions, labels)
        total_loss = loss_dict['total']
        
        # Comprehensive logging for TensorBoard
        self.log('train/total_loss', total_loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('train/bbox_loss', loss_dict['bbox'], sync_dist=True, on_step=True, on_epoch=True)
        self.log('train/obj_loss', loss_dict['obj'], sync_dist=True, on_step=True, on_epoch=True)
        self.log('train/cls_loss', loss_dict['cls'], sync_dist=True, on_step=True, on_epoch=True)
        
        # Learning rate logging
        self.log('train/learning_rate', self.optimizers().param_groups[0]['lr'], sync_dist=True, on_step=True)
        
        # Store for epoch end logging
        self.train_losses.append(total_loss.item())
        
        return {
            'loss': total_loss,
            'bbox_loss': loss_dict['bbox'],
            'obj_loss': loss_dict['obj'],
            'cls_loss': loss_dict['cls']
        }
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, labels = batch
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            predictions = self(images)
            loss_dict = self.loss_fn(predictions, labels)
            total_loss = loss_dict['total']
        
        # Comprehensive validation logging for TensorBoard
        self.log('val/total_loss', total_loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log('val/bbox_loss', loss_dict['bbox'], sync_dist=True, on_step=True, on_epoch=True)
        self.log('val/obj_loss', loss_dict['obj'], sync_dist=True, on_step=True, on_epoch=True)
        self.log('val/cls_loss', loss_dict['cls'], sync_dist=True, on_step=True, on_epoch=True)
        
        # Store for epoch end logging
        self.val_losses.append(total_loss.item())
        
        return {
            'val_loss': total_loss,
            'val_bbox_loss': loss_dict['bbox'],
            'val_obj_loss': loss_dict['obj'],
            'val_cls_loss': loss_dict['cls']
        }
    
    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        if self.train_losses:
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            min_loss = min(self.train_losses)
            max_loss = max(self.train_losses)
            
            # Log comprehensive epoch metrics
            self.log('train/epoch_avg_loss', avg_loss, sync_dist=True)
            self.log('train/epoch_min_loss', min_loss, sync_dist=True)
            self.log('train/epoch_max_loss', max_loss, sync_dist=True)
            self.log('train/epoch_loss_std', np.std(self.train_losses), sync_dist=True)
            logger.info(f"Epoch {self.current_epoch} - Average training loss: {avg_loss:.4f}")
            self.train_losses.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        if self.val_losses:
            avg_loss = sum(self.val_losses) / len(self.val_losses)
            min_loss = min(self.val_losses)
            max_loss = max(self.val_losses)
            
            # Log comprehensive epoch metrics
            self.log('val/epoch_avg_loss', avg_loss, sync_dist=True)
            self.log('val/epoch_min_loss', min_loss, sync_dist=True)
            self.log('val/epoch_max_loss', max_loss, sync_dist=True)
            self.log('val/epoch_loss_std', np.std(self.val_losses), sync_dist=True)
            logger.info(f"Epoch {self.current_epoch} - Average validation loss: {avg_loss:.4f}")
            self.val_losses.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        # AdamW optimizer with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0005,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'frequency': 1
            }
        }
    
    def configure_callbacks(self) -> list:
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='yolov8n-{epoch:02d}-{val_total_loss:.4f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor='val/total_loss',
            patience=10,
            mode='min',
            min_delta=0.001
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def configure_loggers(self) -> list:
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir='logs',
            name='yolov8n_training',
            version=None
        )
        
        return [tb_logger]
    
    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calculate model size
        param_size = 0
        buffer_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'num_classes': self.num_classes,
            'input_size': self.input_size
        }

    def on_train_start(self) -> None:
        """Log model information at training start."""
        model_info = self.get_model_info()
        
        # Log model architecture info
        self.logger.experiment.add_text(
            'model_info',
            f"Total Parameters: {model_info['total_parameters']:,}\n"
            f"Trainable Parameters: {model_info['trainable_parameters']:,}\n"
            f"Model Size: {model_info['model_size_mb']:.2f} MB\n"
            f"Input Size: {model_info['input_size']}x{model_info['input_size']}\n"
            f"Number of Classes: {model_info['num_classes']}"
        )
        
        logger.info(f"Training started with model: {model_info['total_parameters']:,} parameters")



def create_lightning_module(num_classes: int = 3, input_size: int = 640, 
                          learning_rate: float = 0.01) -> YOLOv8nLightningModule:

    return YOLOv8nLightningModule(
        num_classes=num_classes,
        input_size=input_size,
        learning_rate=learning_rate
    )
