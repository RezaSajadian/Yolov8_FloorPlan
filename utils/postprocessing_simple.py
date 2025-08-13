"""
Simplified Post-Processing Module

Clean, working post-processing with:
- Confidence thresholding
- Non-Maximum Suppression (NMS)
- Coordinate decoding
- Duplicate removal
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import cv2

class SimpleYOLOPostProcessor:
    def __init__(self, num_classes: int = 3, confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.3, input_size: int = 640):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.class_names = ['room', 'window', 'door']
        
    def process_predictions(self, predictions: List[torch.Tensor], 
                          original_image_shape: Tuple[int, int]) -> List[Dict]:

        all_detections = []
        
        for scale_idx, pred in enumerate(predictions):
            # pred shape: [batch, channels, height, width]
            batch_size, channels, grid_h, grid_w = pred.shape
            
            # Reshape to [batch, anchors, height, width, channels_per_anchor]
            num_anchors = 3
            channels_per_anchor = 4 + 1 + self.num_classes  # bbox + objectness + classes
            pred_reshaped = pred.view(batch_size, num_anchors, channels_per_anchor, grid_h, grid_w)
            pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2)  # [batch, anchors, height, width, channels]
            
            for anchor_idx in range(num_anchors):
                # Extract predictions for this anchor
                anchor_pred = pred_reshaped[0, anchor_idx]  # [height, width, channels]
                
                # Convert to numpy for easier processing
                objectness = torch.sigmoid(anchor_pred[..., 4]).detach().cpu().numpy()
                class_logits = anchor_pred[..., 5:5+self.num_classes].detach().cpu().numpy()
                class_probs = torch.softmax(torch.tensor(class_logits), dim=-1).numpy()
                bbox_pred = anchor_pred[..., :4].detach().cpu().numpy()
                
                # Find high confidence cells
                high_conf_indices = np.where(objectness > self.confidence_threshold)
                
                for y_idx, x_idx in zip(high_conf_indices[0], high_conf_indices[1]):
                    # Get predictions for this cell
                    cell_objectness = objectness[y_idx, x_idx]
                    cell_class_probs = class_probs[y_idx, x_idx]
                    cell_bbox = bbox_pred[y_idx, x_idx]
                    
                    # Get best class
                    best_class = np.argmax(cell_class_probs)
                    best_class_prob = cell_class_probs[best_class]
                    
                    # Calculate confidence score
                    confidence = cell_objectness * best_class_prob
                    
                    if confidence > self.confidence_threshold:
                        # Decode bounding box coordinates
                        bbox = self._decode_bbox(cell_bbox, x_idx, y_idx, grid_w, grid_h)
                        
                        # Convert to original image coordinates
                        img_h, img_w = original_image_shape
                        bbox[0] *= img_w  # x1
                        bbox[1] *= img_h  # y1
                        bbox[2] *= img_w  # x2
                        bbox[3] *= img_h  # y2
                        
                        detection = {
                            'bbox': bbox,  # [x1, y1, x2, y2]
                            'confidence': confidence,
                            'class_id': best_class,
                            'class_name': self.class_names[best_class]
                        }
                        
                        all_detections.append(detection)
        
        # Apply NMS
        if all_detections:
            all_detections = self._apply_nms(all_detections)
            
        return all_detections
    
    def _decode_bbox(self, bbox_pred: np.ndarray, x_idx: int, y_idx: int, 
                     grid_w: int, grid_h: int) -> List[float]:

        # Normalize grid coordinates
        x_center = (x_idx + 0.5) / grid_w
        y_center = (y_idx + 0.5) / grid_h
        
        # Decode width and height
        w = 1.0 / (1.0 + np.exp(-bbox_pred[2])) * 0.3  # Scale down
        h = 1.0 / (1.0 + np.exp(-bbox_pred[3])) * 0.3  # Scale down
        
        # Ensure minimum size
        w = max(w, 0.05)
        h = max(h, 0.05)
        
        # Convert to x1, y1, x2, y2 format
        x1 = max(0.0, x_center - w/2)
        y1 = max(0.0, y_center - h/2)
        x2 = min(1.0, x_center + w/2)
        y2 = min(1.0, y_center + h/2)
        
        return [x1, y1, x2, y2]
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:

        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Group by class
        class_groups = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(det)
        
        # Apply NMS for each class
        final_detections = []
        for class_id, class_dets in class_groups.items():
            if len(class_dets) == 1:
                final_detections.append(class_dets[0])
            else:
                # Apply NMS for this class
                class_dets = self._nms_single_class(class_dets)
                final_detections.extend(class_dets)
        
        return final_detections
    
    def _nms_single_class(self, detections: List[Dict]) -> List[Dict]:
        
        if len(detections) <= 1:
            return detections
        
        # Convert to numpy for easier computation
        bboxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Apply OpenCV NMS
        try:
            indices = cv2.dnn.NMSBoxes(
                bboxes.tolist(), 
                scores.tolist(), 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            else:
                return []
        except Exception:
            # Fallback: manual NMS
            return self._manual_nms(detections)
    
    def _manual_nms(self, detections: List[Dict]) -> List[Dict]:
        # Manual NMS implementation as fallback.
        
        if len(detections) <= 1:
            return detections
        
        kept = []
        for detection in detections:
            should_keep = True
            
            for kept_det in kept:
                iou = self._calculate_iou(detection['bbox'], kept_det['bbox'])
                if iou > self.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(detection)
        
        return kept
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        # Calculate Intersection over Union between two bounding boxes.

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
