#!/usr/bin/env python3

import os
import shutil
import random
from pathlib import Path
import yaml
from typing import List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> dict:
    # Get the project root directory (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'configs' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_all_data_files(data_dir: str) -> Tuple[List[str], List[str]]:
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError(f"Images or labels directory not found in {data_dir}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
    
    # Get corresponding label files
    label_files = []
    for img_path in image_files:
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            label_files.append(str(label_path))
            image_files[image_files.index(img_path)] = str(img_path)
        else:
            logger.warning(f"No label file found for {img_path}")
    
    # Filter to only keep pairs that have both image and label
    valid_pairs = []
    for img_path, label_path in zip(image_files, label_files):
        if os.path.exists(img_path) and os.path.exists(label_path):
            valid_pairs.append((img_path, label_path))
    
    logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
    return valid_pairs

def split_dataset(data_pairs: List[Tuple[str, str]], 
                 train_split: float, 
                 val_split: float, 
                 test_split: float,
                 random_seed: int = 42) -> Tuple[List, List, List]:
    
    # Validate splits
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Split proportions must sum to 1.0, got {total_split}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle data pairs
    shuffled_pairs = data_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    total_samples = len(shuffled_pairs)
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    
    # Split the data
    train_data = shuffled_pairs[:train_size]
    val_data = shuffled_pairs[train_size:train_size + val_size]
    test_data = shuffled_pairs[train_size + val_size:]
    
    logger.info(f"Dataset split:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Train: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    logger.info(f"  Val: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")
    logger.info(f"  Test: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")
    
    return train_data, val_data, test_data

def copy_data_to_split(data_pairs: List[Tuple[str, str]], 
                      target_dir: str, 
                      split_name: str):
    
    # Create directories
    images_dir = os.path.join(target_dir, 'images')
    labels_dir = os.path.join(target_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Copy files
    for img_path, label_path in data_pairs:
        # Copy image
        img_filename = os.path.basename(img_path)
        img_dest = os.path.join(images_dir, img_filename)
        shutil.copy2(img_path, img_dest)
        
        # Copy label
        label_filename = os.path.basename(label_path)
        label_dest = os.path.join(labels_dir, label_filename)
        shutil.copy2(label_path, label_dest)
    
    logger.info(f"Copied {len(data_pairs)} files to {split_name} split")

def main():
    logger.info("Starting dataset splitting...")
    
    # Load configuration
    config = load_config()
    data_config = config['data']
    
    # Get split proportions
    train_split = data_config['train_split']
    val_split = data_config['val_split']
    test_split = data_config['test_split']
    
    # Source data directory (assuming all data is in one place)
    source_data_dir = "data"  # This should contain images/ and labels/ subdirectories
    
    # Target directories
    train_dir = data_config['train_path']
    val_dir = data_config['val_path']
    test_dir = data_config['test_path']
    
    logger.info(f"Source data directory: {source_data_dir}")
    logger.info(f"Train directory: {train_dir}")
    logger.info(f"Val directory: {val_dir}")
    logger.info(f"Test directory: {test_dir}")
    
    try:
        # Get all data files
        data_pairs = get_all_data_files(source_data_dir)
        
        if len(data_pairs) == 0:
            logger.error("No valid data pairs found!")
            return
        
        # Split the dataset
        train_data, val_data, test_data = split_dataset(
            data_pairs, train_split, val_split, test_split
        )
        
        # Clear existing directories
        for target_dir in [train_dir, val_dir, test_dir]:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
                logger.info(f"Cleared existing directory: {target_dir}")
        
        # Copy data to split directories
        copy_data_to_split(train_data, train_dir, "train")
        copy_data_to_split(val_data, val_dir, "validation")
        copy_data_to_split(test_data, test_dir, "test")
        
        # Create data.yaml for YOLO format
        data_yaml = {
            'path': 'data_organized',
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': config['model']['num_classes'],
            'names': ['room', 'window', 'door']
        }
        
        yaml_path = os.path.join('data_organized', 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml at {yaml_path}")
        logger.info("Dataset splitting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}")
        raise

if __name__ == "__main__":
    main()
