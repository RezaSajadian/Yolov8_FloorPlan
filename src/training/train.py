#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.yolo_model import create_yolov8n_model
# Loss import removed - using working_loss directly
from src.data.dataset import FloorplanDataset
from src.training.lightning_module import create_lightning_module
from src.utils.logger import get_logger
from src.utils.config import config_manager

logger = get_logger(__name__)


# Custom collate function to handle variable number of labels
def collate_fn(batch):
    # Collate function to handle variable number of labels in a batch.
    import torch
    images = torch.stack([item[0] for item in batch])
    
    # Find maximum number of labels in this batch
    max_labels = max(len(item[1]) for item in batch)
    
    # Pad all label tensors to the same size
    padded_labels = []
    for item in batch:
        labels = item[1]
        if len(labels) < max_labels:
            # Pad with zeros
            padding = torch.zeros((max_labels - len(labels), 5))
            padded = torch.cat([labels, padding], dim=0)
            padded_labels.append(padded)
        else:
            padded_labels.append(labels)
    
    labels = torch.stack(padded_labels)
    return images, labels


def setup_training_environment():
    logger.info("Setting up training environment...")
    
    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Get configuration sections
    model_config = config_manager.get_model_config()
    data_config = config_manager.get_data_config()
    training_config = config_manager.get_training_config()
    hardware_config = config_manager.get_hardware_config()
    
    # Create necessary directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    logger.info("Training environment setup complete")
    
    return {
        'model': model_config,
        'data': data_config,
        'training': training_config,
        'hardware': hardware_config
    }


def create_data_loaders(config: dict):
    logger.info("Creating data loaders...")
    
    # Get data paths and proportions from config
    train_path = config['data'].get('train_path', 'data/images')
    val_path = config['data'].get('val_path', 'data/images')
    train_proportion = config['data'].get('train_data_proportion', 1.0)
    val_proportion = config['data'].get('val_data_proportion', 0.2)
    
    logger.info(f"Training data proportion: {train_proportion*100:.1f}%")
    logger.info(f"Validation data proportion: {val_proportion*100:.1f}%")
    
    # Check if data directories exist
    if not Path(train_path).exists():
        logger.warning(f"Training data directory not found: {train_path}")
        logger.info("Creating placeholder dataset for demonstration")
        # For now, we'll create empty datasets - will be populated when data is available
        train_dataset = FloorplanDataset(
            data_dir=train_path,
            img_size=config['model']['input_size'],
            augment=True,
            is_training=True,
            data_proportion=train_proportion
        )
    else:
        train_dataset = FloorplanDataset(
            data_dir=train_path,
            img_size=config['model']['input_size'],
            augment=True,
            is_training=True,
            data_proportion=train_proportion
        )
    
    if not Path(val_path).exists():
        logger.warning(f"Validation data directory not found: {val_path}")
        logger.info("Using training data for validation (not recommended for production)")
        val_dataset = train_dataset
    else:
        val_dataset = FloorplanDataset(
            data_dir=val_path,
            img_size=config['model']['input_size'],
            augment=False,
            is_training=False,
            data_proportion=val_proportion
        )
    
    # Create data loaders
    if len(train_dataset) == 0:
        logger.error("No training data found. Please add images to data/train/ directory.")
        logger.info("Example data structure:")
        logger.info("  data/train/floorplan1.jpg")
        logger.info("  data/train/floorplan1.txt")
        logger.info("  data/train/floorplan2.jpg")
        logger.info("  data/train/floorplan2.txt")
        sys.exit(1)
    

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware'].get('num_workers', 0),
        pin_memory=config['hardware'].get('pin_memory', False),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware'].get('num_workers', 0),
        pin_memory=config['hardware'].get('pin_memory', False),
        collate_fn=collate_fn
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    return train_loader, val_loader


def setup_trainer(config: dict):

    logger.info("Setting up PyTorch Lightning trainer...")
    
    # Callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='yolov8n-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('patience', 10),
            mode='min',
            min_delta=config['training'].get('min_delta', 0.001)
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger - make TensorBoard optional
    try:
        logger_inst = TensorBoardLogger(
            save_dir='logs',
            name='yolov8n_training',
            version=None
        )
        logger.info("TensorBoard logger initialized")
    except Exception as e:
        logger.warning(f"TensorBoard not available, using basic logging: {e}")
        logger_inst = None
    
    # Trainer configuration
    trainer_kwargs = {
        'max_epochs': config['training']['num_epochs'],
        'callbacks': callbacks,
        'logger': logger_inst,
        'accelerator': 'auto',  # Automatically detect device
        'devices': config['hardware'].get('num_gpus', 1),
        'precision': 16 if config['hardware'].get('mixed_precision', True) else 32,
        'gradient_clip_val': 0.1,
        'accumulate_grad_batches': config['hardware'].get('gradient_accumulation_steps', 1),
        'log_every_n_steps': config.get('logging', {}).get('log_every_n_steps', 10),
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': False,
        'reload_dataloaders_every_n_epochs': 0
    }
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    logger.info("Trainer setup complete")
    return trainer


def main(config_path: str = 'configs/config.yaml', resume_path: str = None, 
         epochs: int = None, batch_size: int = None, learning_rate: float = None):
    """Main training function with CLI override support."""
    try:
        # Setup training environment
        config = setup_training_environment()
        
        # Apply CLI overrides to config
        if epochs is not None:
            config['training']['num_epochs'] = epochs
            logger.info(f"CLI override: epochs = {epochs}")
        if batch_size is not None:
            config['training']['batch_size'] = batch_size
            logger.info(f"CLI override: batch_size = {batch_size}")
        if learning_rate is not None:
            config['training']['learning_rate'] = learning_rate
            logger.info(f"CLI override: learning_rate = {learning_rate}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(config)
        
        # Create Lightning module
        lightning_module = create_lightning_module(
            num_classes=config['model']['num_classes'],
            input_size=config['model']['input_size'],
            learning_rate=config['training']['learning_rate']
        )
        
        # Log model information
        model_info = lightning_module.get_model_info()
        logger.info(f"Model parameters: {model_info['total_parameters']:,}")
        logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Setup trainer
        trainer = setup_trainer(config)
        
        # Start training
        logger.info("Starting training...")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_path
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        import torch
        final_model_path = "checkpoints/yolov8n_final.pt"
        torch.save(lightning_module.model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    # Parse CLI arguments for overrides
    parser = argparse.ArgumentParser(description='Train YOLOv8n model with config-first approach')
    parser.add_argument('--epochs', type=int, help='Override number of epochs from config')
    parser.add_argument('--batch-size', type=int, help='Override batch size from config')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate from config')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Call main with CLI overrides
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_path=args.resume
    )
