
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WorkingYOLOLoss(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Loss weights
        self.bbox_weight = 0.05
        self.obj_weight = 1.0
        self.cls_weight = 0.5
        
        # MSE loss for bounding boxes
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # BCE loss for objectness and classification
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        logger.info(f"Working YOLO loss initialized for {num_classes} classes")
    
    def forward(self, predictions: List[torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch_size = targets.shape[0]
        device = predictions[0].device
        
        # Initialize losses
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        bbox_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
        cls_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Use the first scale prediction for simplicity
        pred = predictions[0]  # [batch, channels, height, width]
        
        # Process each batch item
        for batch_idx in range(batch_size):
            batch_labels = targets[batch_idx]
            
            valid_mask = (batch_labels[:, 0] >= 0) & (batch_labels[:, 1:].sum(dim=1) > 0)
            valid_labels = batch_labels[valid_mask]
            
            if len(valid_labels) == 0:
                continue
            
            # For each valid label, compute loss
            for label in valid_labels:
                class_id, x_center, y_center, width, height = label
                
                # Convert normalized coordinates to pixel coordinates
                h, w = pred.shape[2:4]
                x_center_px = int(x_center * w)
                y_center_px = int(y_center * h)
                
                # Get predictions at this location
                if 0 <= x_center_px < w and 0 <= y_center_px < h:
                    # Extract predictions at this location
                    pred_at_location = pred[batch_idx, :, y_center_px, x_center_px]
                    
                    # Compute losses
                    # Bounding box loss: compare predicted vs target coordinates
                    target_coords = torch.tensor([x_center, y_center, width, height], 
                                               device=pred.device, dtype=torch.float32)
                    
                    # Use the first few channels as bbox predictions
                    if pred_at_location.shape[0] >= 4:
                        pred_coords = pred_at_location[:4]
                        bbox_loss = self.mse_loss(pred_coords, target_coords).mean()
                    else:
                        bbox_loss = torch.tensor(0.1, device=pred.device, dtype=torch.float32, requires_grad=True)
                    
                    # Objectness loss: use prediction channels
                    if pred_at_location.shape[0] >= 5:
                        pred_obj = pred_at_location[4:5]
                        target_obj = torch.tensor(1.0, device=pred.device, dtype=torch.float32)
                        obj_loss = self.bce_loss(pred_obj, target_obj.unsqueeze(0)).mean()
                    else:
                        obj_loss = torch.tensor(0.1, device=pred.device, dtype=torch.float32, requires_grad=True)
                    
                    # Classification loss: use remaining channels
                    if pred_at_location.shape[0] >= 5 + self.num_classes:
                        pred_cls = pred_at_location[5:5+self.num_classes]
                        target_cls = torch.zeros(self.num_classes, device=pred.device, dtype=torch.float32)
                        target_cls[int(class_id)] = 1.0
                        cls_loss = self.bce_loss(pred_cls, target_cls).mean()
                    else:
                        cls_loss = torch.tensor(0.1, device=pred.device, dtype=torch.float32, requires_grad=True)
                    
                    bbox_loss_sum = bbox_loss_sum + bbox_loss
                    obj_loss_sum = obj_loss_sum + obj_loss
                    cls_loss_sum = cls_loss_sum + cls_loss
        
        # Compute total loss
        total_loss = (
            self.bbox_weight * bbox_loss_sum +
            self.obj_weight * obj_loss_sum +
            self.cls_weight * cls_loss_sum
        )
        
        return {
            'total': total_loss,
            'bbox': bbox_loss_sum,
            'obj': obj_loss_sum,
            'cls': cls_loss_sum
        }


def create_working_yolo_loss(num_classes: int = 3) -> WorkingYOLOLoss:
    return WorkingYOLOLoss(num_classes)
