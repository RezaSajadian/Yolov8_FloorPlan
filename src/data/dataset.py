import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FloorplanDataset(Dataset):    
    def __init__(self, data_dir: str, img_size: int = 640, subset_ratio: float = 1.0,
                 augment: bool = True, is_training: bool = True, data_proportion: float = 1.0):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.subset_ratio = subset_ratio
        self.data_proportion = data_proportion
        self.augment = augment and is_training
        self.is_training = is_training
        
        # Class names for floorplan detection
        self.class_names = ['room', 'window', 'door']
        self.num_classes = len(self.class_names)
        
        # Load file paths
        self.img_files, self.label_files = self._load_file_paths()
        
        # Apply data proportion for training/validation
        if data_proportion < 1.0 and len(self.img_files) > 0:
            num_samples = max(1, int(len(self.img_files) * data_proportion))
            indices = random.sample(range(len(self.img_files)), min(num_samples, len(self.img_files)))
            self.img_files = [self.img_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
            logger.info(f"Using {data_proportion*100:.1f}% of data: {len(self.img_files)} images")
        
        # Apply subset ratio for fast inference (additional reduction)
        if subset_ratio < 1.0 and len(self.img_files) > 0:
            num_samples = max(1, int(len(self.img_files) * subset_ratio))
            indices = random.sample(range(len(self.img_files)), min(num_samples, len(self.img_files)))
            self.img_files = [self.img_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
            logger.info(f"Using {subset_ratio*100:.1f}% subset: {len(self.img_files)} images")
        
        logger.info(f"Dataset initialized with {len(self.img_files)} images")
    
    def _load_file_paths(self) -> Tuple[List[str], List[str]]:
        img_files = []
        label_files = []
        
        # Check if we have separate images and labels folders
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"
        
        # Also check for test_images and test_labels structure
        if not images_dir.exists():
            images_dir = self.data_dir
            labels_dir = Path(str(self.data_dir).replace("test_images", "test_labels"))
        
        if images_dir.exists() and labels_dir.exists():
            # Separate images and labels folders structure
            logger.info("Found separate images and labels folders")
            
            # Find image files
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in img_extensions:
                img_files.extend(list(images_dir.glob(f"*{ext}")))
                img_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
            
            # Find corresponding labels
            valid_img_files = []
            for img_path in img_files:
                # Get the base name without extension
                base_name = img_path.stem
                label_path = labels_dir / f"{base_name}.txt"
                
                if label_path.exists():
                    valid_img_files.append(str(img_path))
                    label_files.append(str(label_path))
            
            if not valid_img_files:
                logger.warning(f"No valid image-label pairs found in {images_dir} and {labels_dir}")
                return [], []
            
            logger.info(f"Found {len(valid_img_files)} valid image-label pairs")
            return valid_img_files, label_files
        
        else:
            # Single folder structure (original logic)
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in img_extensions:
                img_files.extend(list(self.data_dir.glob(f"*{ext}")))
                img_files.extend(list(self.data_dir.glob(f"*{ext.upper()}")))
            
            # Find corresponding labels
            valid_img_files = []
            for img_path in img_files:
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    valid_img_files.append(str(img_path))
                    label_files.append(str(label_path))
            
            if not valid_img_files:
                logger.warning(f"No valid image-label pairs found in {self.data_dir}")
                return [], []
            
            logger.info(f"Found {len(valid_img_files)} valid image-label pairs")
            return valid_img_files, label_files
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.img_files[idx]
        img = self._load_image(img_path)
        
        # Load labels
        label_path = self.label_files[idx]
        labels = self._load_labels(label_path) if os.path.exists(label_path) else []
        
        # Apply augmentations
        if self.augment:
            img, labels = self._apply_augmentations(img, labels)
        
        # Resize and adjust labels
        img, labels = self._resize_image_and_labels(img, labels)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        if labels:
            # Convert labels list to numpy array
            labels_array = np.array(labels, dtype=np.float32)
            label_tensor = torch.from_numpy(labels_array).float()
        else:
            label_tensor = torch.zeros((0, 5))
        
        return img_tensor, label_tensor
    
    def _load_image(self, img_path: str) -> np.ndarray:
        # Load image from file with better PNG handling.
        try:
            # Use PIL for better PNG handling (fixes iCCP warnings)
            from PIL import Image
            import warnings
            
            # Suppress iCCP warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
            
            # Load with PIL first
            pil_img = Image.open(img_path).convert('RGB')
            img = np.array(pil_img)
            
            if img is None or img.size == 0:
                raise ValueError(f"Could not load image or image is empty: {img_path}")
            
            # Check if image is too small
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Image too small: {img_path} - {img.shape}")
                
            return img
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
    
    def _load_labels(self, label_path: str) -> List[List[float]]:
        # Load YOLO format labels
        try:
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(x) for x in line.split()]
                        if len(values) == 5:
                            labels.append(values)
            return labels
        except Exception as e:
            logger.error(f"Error loading labels {label_path}: {e}")
            return []
    
    def _apply_augmentations(self, img: np.ndarray, labels: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        if not self.augment or not labels:
            return img, labels
        
        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            for label in labels:
                label[1] = 1.0 - label[1]  # Flip x_center
        
        return img, labels
    
    def _resize_image_and_labels(self, img: np.ndarray, labels: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        h, w = img.shape[:2]
        
        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Adjust labels
        adjusted_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # Scale coordinates
            x_center = x_center * self.img_size / w
            y_center = y_center * self.img_size / h
            width = width * self.img_size / w
            height = height * self.img_size / h
            
            # Clamp to boundaries
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Keep visible labels
            if width > 0.01 and height > 0.01:
                adjusted_labels.append([class_id, x_center, y_center, width, height])
        
        return img_resized, adjusted_labels
    
    def get_class_names(self) -> List[str]:
        return self.class_names
    
    def get_num_classes(self) -> int:
        return self.num_classes


def create_floorplan_dataset(data_dir: str, img_size: int = 640, subset_ratio: float = 1.0,
                           augment: bool = True, is_training: bool = True) -> FloorplanDataset:
    return FloorplanDataset(
        data_dir=data_dir,
        img_size=img_size,
        subset_ratio=subset_ratio,
        augment=augment,
        is_training=is_training
    )
