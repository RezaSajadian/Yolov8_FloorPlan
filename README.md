# YOLOv8n Floorplan Object Detection

An implementation of YOLOv8n for detecting rooms, windows, and doors in architectural floorplans. Built from scratch using PyTorch and PyTorch Lightning.

## What This Project Does

- **Detects 3 classes**: rooms, windows, doors
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes with confidence scores
- **Model size**: ~2.9 MB, ~744K parameters

## Quick Start

### 1. Setup Environment
```bash
# Create and activate Python environment (recommended)
conda create -n yolov8n python=3.9
conda activate yolov8n

# Or use venv
python -m venv yolov8n_env
source yolov8n_env/bin/activate  # On Windows: yolov8n_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup project structure (creates necessary folders)
python setup_project.py
```

**What `setup_project.py` does:**
- Creates `data/`, `data_organized/`, `checkpoints/`, `logs/`, `outputs/` folders
- Copies `config_template.yaml` to `config.yaml`
- Creates a guide for dataset organization

### 2. Prepare Your Dataset

#### **Option A: Use Your Own Floorplan Data**
```bash
# Place your images and labels in:
data/
├── images/          # Your floorplan images (.jpg, .png)
└── labels/          # YOLO format labels (.txt)
```

#### **Option B: Create Sample Data for Testing**
```bash
# Create a simple test image and label
mkdir -p data/images data/labels

# Create a simple test image (640x640 white rectangle)
python -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 640), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 300, 300], outline='black', width=2)
img.save('data/images/test_floorplan.jpg')
"

# Create corresponding YOLO label file
echo "0 0.3125 0.3125 0.3125 0.3125" > data/labels/test_floorplan.txt
```

#### **YOLO Label Format Explained**
Each line in a `.txt` file represents one object:
```
class_id x_center y_center width height
```

**Class IDs:**
- `0` = room
- `1` = window  
- `2` = door

**Coordinates are normalized (0.0 to 1.0):**
- `x_center, y_center`: Center point of bounding box
- `width, height`: Width and height of bounding box

**Example label file (`data/labels/test_floorplan.txt`):**
```
0 0.3125 0.3125 0.3125 0.3125
1 0.8 0.2 0.1 0.15
2 0.2 0.8 0.15 0.1
```

This means:
- A room (class 0) at center (200, 200) with size 200x200 pixels
- A window (class 1) at center (512, 128) with size 64x96 pixels  
- A door (class 2) at center (128, 512) with size 96x64 pixels

### 3. Organize Dataset
```bash
# Automatically split into train/val/test
python src/data/split_dataset.py
```

**What this script does:**
- Reads from `data/images/` and `data/labels/`
- Splits into 70% train, 10% validation, 20% test
- Creates organized structure in `data_organized/`

**Expected output:**
```
data_organized/
├── train/
│   ├── images/     # 70% of your images
│   └── labels/     # 70% of your labels
├── val/
│   ├── images/     # 10% of your images
│   └── labels/     # 10% of your labels
└── test/
    ├── images/     # 20% of your images
    └── labels/     # 20% of your labels
```

**If the script fails:**
- Check that `data/images/` and `data/labels/` exist
- Ensure you have at least 3 images for splitting
- Check file permissions

### 4. Configure Training
```bash
# Copy and edit configuration
cp configs/config_template.yaml configs/config.yaml

# Edit the configuration file
nano configs/config.yaml  # or use your preferred editor
```

**Complete configuration example:**
```yaml
data:
  train_path: "data_organized/train"
  val_path: "data_organized/val"
  test_path: "data_organized/test"
  
  # Dataset splitting (already done by split_dataset.py)
  train_split: 0.70
  val_split: 0.10
  test_split: 0.20
  
  # How much of each split to use (1.0 = use all)
  train_data_proportion: 1.0
  val_data_proportion: 1.0
  test_data_proportion: 1.0

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0003
  
  loss_weights:
    bbox: 0.1
    obj: 0.7
    cls: 0.6

evaluation:
  model_path: "checkpoints/your_checkpoint_name.pt"  # Update this after training
  max_eval_images: 10
  confidence_threshold: 0.25
  nms_threshold: 0.45
  batch_size: 8

inference:
  model_path: "checkpoints/your_checkpoint_name.pt"  # Update this after training
  test_image: "data_organized/test/images/your_test_image.jpg"

hardware:
  device: "auto"
  num_workers: 0
  pin_memory: false
```

### 5. Test Run (Quick Validation)
```bash
# Test with minimal data to ensure everything works
# Edit configs/config.yaml first:
data:
  train_data_proportion: 0.1  # Use only 10% of data
  val_data_proportion: 0.1    # Use only 10% of data

training:
  num_epochs: 3               # Just 3 epochs for testing
  batch_size: 16              # Smaller batch size

# Run quick training test
python src/training/train.py

# This should complete in under 5 minutes
# Check that checkpoints/ folder is created
ls checkpoints/
```

### 6. Full Training
```bash
# Reset config to full training
# Edit configs/config.yaml:
data:
  train_data_proportion: 1.0  # Use all training data
  val_data_proportion: 1.0    # Use all validation data

training:
  num_epochs: 50              # Full training
  batch_size: 32              # Full batch size

# Start full training
python src/training/train.py

# Training will:
# - Save checkpoints every 5 epochs
# - Stop early if no improvement
# - Export ONNX model when done
```

### 7. Evaluate Model
```bash
# First, check what checkpoints you have
ls checkpoints/

# Then evaluate using the correct checkpoint name
python src/evaluation/model_evaluator.py --model-path checkpoints/your_checkpoint_name.pt

# Or update the config file first:
# Edit configs/config.yaml and set:
evaluation:
  model_path: "checkpoints/your_checkpoint_name.pt"

# Then run without arguments
python src/evaluation/model_evaluator.py
```

**This generates:**
- Performance metrics (mAP, precision, recall)
- Bounding box images in `evaluation_results/bbox_images/`
- Performance comparison plots

### 8. Run Inference
```bash
# First, check what checkpoints you have
ls checkpoints/

# Update configs/config.yaml with your test image:
inference:
  model_path: "checkpoints/your_checkpoint_name.pt"
  test_image: "data_organized/test/images/your_test_image.jpg"

# Run inference
python src/inference/inference_engine.py
```

## Configuration Options

### Data Proportions
```yaml
data:
  # How much of your total dataset to use
  train_split: 0.70    # 70% for training
  val_split: 0.10      # 10% for validation
  test_split: 0.20     # 20% for testing
  
  # How much of each split to actually use
  train_data_proportion: 1.0   # Use 100% of train split
  val_data_proportion: 1.0     # Use 100% of val split
  test_data_proportion: 1.0    # Use 100% of test split
```

### Training Parameters
```yaml
training:
  batch_size: 32              # Batch size (reduce if OOM)
  num_epochs: 50              # Total training epochs
  learning_rate: 0.0003       # Learning rate
  weight_decay: 0.0003        # L2 regularization
  
  # Loss weights
  loss_weights:
    bbox: 0.1                 # Bounding box loss
    obj: 0.7                  # Objectness loss
    cls: 0.6                  # Classification loss
```

### Evaluation Settings
```yaml
evaluation:
  max_eval_images: 10         # Number of test images to evaluate
  confidence_threshold: 0.25  # Detection confidence threshold
  nms_threshold: 0.45        # Non-maximum suppression threshold
  batch_size: 8               # Evaluation batch size
```

### Hardware Settings
```yaml
hardware:
  device: "auto"              # auto, cpu, cuda, mps
  num_workers: 0              # Set to 0 for macOS
  pin_memory: false           # Set to false for macOS
```

## Project Structure
```
src/
├── models/                   # YOLOv8n model architecture
├── data/                     # Dataset loading and preprocessing
├── training/                 # Training scripts and PyTorch Lightning
├── inference/                # Model inference and prediction
├── evaluation/               # Model evaluation and metrics
└── utils/                    # Configuration, logging, post-processing

configs/
├── config_template.yaml      # Template configuration (safe to commit)
└── config.yaml              # Your configuration (don't commit)

data_organized/               # Your organized dataset
├── train/                    # Training images and labels
├── val/                      # Validation images and labels
└── test/                     # Test images and labels
```

## Common Issues and Solutions

### Setup Issues
```bash
# If setup_project.py fails
mkdir -p data data_organized checkpoints logs outputs evaluation_results
cp configs/config_template.yaml configs/config.yaml

# If split_dataset.py fails
ls data/images/ data/labels/  # Check files exist
python src/data/split_dataset.py --help  # Check script options
```

### Memory Issues
```yaml
# Reduce batch size
training:
  batch_size: 16

# Reduce data usage
data:
  train_data_proportion: 0.5  # Use only 50% of training data
```

### Slow Training
```yaml
# Reduce number of epochs
training:
  num_epochs: 20

# Use less data for quick testing
data:
  train_data_proportion: 0.1  # Use only 10% of training data
```

### No Detections
```yaml
# Lower confidence threshold
evaluation:
  confidence_threshold: 0.1

# Adjust NMS threshold
evaluation:
  nms_threshold: 0.6
```

### Import Errors
```bash
# If you get import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or run from project root directory
```

## What Gets Generated

### Training Outputs
- `checkpoints/` - Model weights and checkpoints
- `logs/` - Training logs and TensorBoard files
- `models/yolov8n.onnx` - ONNX export for deployment

### Evaluation Outputs
- `evaluation_results/` - Performance metrics and plots
- `evaluation_results/bbox_images/` - Images with bounding boxes
- `evaluation_results/evaluation_results.json` - Detailed metrics

### Inference Outputs
- `outputs/` - Predicted images with bounding boxes

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended for training

## Support

- Check the configuration file for all available options
- Read error messages carefully - they usually tell you what's wrong
- Reduce batch size or data usage if you run out of memory
- Use smaller datasets for testing before running on full data
- If you get stuck, check the `logs/` folder for error details
