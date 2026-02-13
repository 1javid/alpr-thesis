# ALPR Object Detection System

A modular, multi-model object detection framework for Automatic License Plate Recognition (ALPR) research. This system provides a unified pipeline for training and inference across multiple state-of-the-art object detection architectures.

## 🎯 Features

- **Multi-Model Support**: Train and compare YOLOv11, YOLOv10, and RT-DETRv2
- **Unified Data Pipeline**: Automated dataset merging, preprocessing, and augmentation
- **Format Conversion**: Automatic conversion between YOLO and COCO annotation formats
- **Flexible Training**: Configuration-driven training with model-specific hyperparameters
- **Easy Inference**: Single interface for inference across all supported models
- **Production Ready**: Professional code structure with comprehensive documentation

## 📁 Project Structure

```
alpr-thesis/
├── configs/
│   ├── base_config.yaml      # Master configuration file
│   └── final_data.yaml        # Auto-generated YOLO data config
├── data_engine/
│   ├── prepare.py             # Dataset merging and augmentation
│   ├── converter.py           # Format conversion (YOLO ↔ COCO)
│   └── augmentor.py           # Augmentation pipeline
├── models/
│   ├── base_trainer.py        # Abstract trainer base class
│   ├── yolov11_trainer.py     # YOLOv11 training implementation
│   └── yolov10_trainer.py     # YOLOv10 training implementation
├── train.py                   # Training entry point
├── infer.py                   # Inference entry point
└── requirements.txt           # Project dependencies
```

## 📂 Dataset Structure Requirements

**IMPORTANT**: Before running the data preparation pipeline, ensure your datasets follow the correct structure.

### Required Directory Structure

Both public and private datasets must follow this structure:

```
raw_data/
├── public_dataset/
│   ├── images/
│   │   ├── train/           # Training images
│   │   │   ├── image001.jpg
│   │   │   ├── image002.jpg
│   │   │   └── ...
│   │   └── valid/           # Validation images (or 'val')
│   │       ├── image001.jpg
│   │       └── ...
│   └── labels/
│       ├── train/           # Training labels (YOLO format)
│       │   ├── image001.txt
│       │   ├── image002.txt
│       │   └── ...
│       └── valid/           # Validation labels (or 'val')
│           ├── image001.txt
│           └── ...
│
└── private_dataset/
    ├── images/
    │   ├── train/
    │   ├── valid/           # or 'val'
    │   └── test/            # Optional test split
    └── labels/
        ├── train/
        ├── valid/           # or 'val'
        └── test/            # Optional test split
```

### YOLO Label Format

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```

All values are normalized (0-1):
- `class_id`: Integer class ID (0, 1, 2, ...)
- `x_center, y_center`: Center coordinates (normalized by image width/height)
- `width, height`: Box dimensions (normalized by image width/height)

**Example** (`image001.txt`):
```
0 0.5 0.5 0.3 0.2
0 0.7 0.3 0.15 0.1
```

### Configuration Mapping

Update `configs/base_config.yaml` to match your directory names:

```yaml
datasets:
  public:
    root: "./raw_data/public_dataset"
    train_subdir: "images/train"    # Must match your folder name
    val_subdir: "images/valid"      # Use 'valid' or 'val' as needed
  
  private:
    root: "./raw_data/private_dataset"
    train_subdir: "images/train"
    val_subdir: "images/valid"      # Use 'valid' or 'val' as needed
    test_subdir: "images/test"      # Optional
```

### Key Points:

✅ **Labels must mirror images directory structure**
- If image is in `images/train/`, label must be in `labels/train/`
- Label filename must match image filename (e.g., `img.jpg` → `img.txt`)

✅ **Subdirectory names can vary**
- Common variations: `valid` vs `val`, `train2017` vs `train`
- Just update `base_config.yaml` to match your structure

✅ **File extensions**
- Images: `.jpg`, `.jpeg`, `.png`
- Labels: `.txt` (YOLO format)

⚠️ **Private dataset MUST follow the same structure as public dataset**
- Don't use custom or inconsistent folder structures
- The pipeline expects parallel `images/` and `labels/` directories

## 🚀 Quick Start

### 1. Installation

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git
- (Optional) NVIDIA GPU with CUDA support for faster training

#### Linux/macOS Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0  # Required for OpenCV

# Clone the repository
git clone <repository-url>
cd alpr-thesis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Windows Installation

```bash
# Clone the repository
git clone <repository-url>
cd alpr-thesis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### GPU Setup (Linux)

For NVIDIA GPU acceleration:

```bash
# Check if CUDA is available
nvidia-smi

# Install PyTorch with CUDA support (if not already in requirements.txt)
# Visit https://pytorch.org/get-started/locally/ for the latest command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available in PyTorch
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 2. Prepare Your Datasets

**Before training, ensure your datasets follow the required structure (see above).**

Place your datasets in the `raw_data/` directory:
```bash
mkdir -p raw_data/public_dataset
mkdir -p raw_data/private_dataset

# Copy/move your datasets here following the structure shown above
```

### 3. Configuration

Edit `configs/base_config.yaml` to specify:
- Dataset paths and subdirectory names (must match your actual folder structure)
- Class names and IDs
- Image size and augmentation parameters
- Model-specific hyperparameters

**Example configuration:**
```yaml
datasets:
  public:
    root: "./raw_data/public_dataset"
    train_subdir: "images/train"     # ← Must match your folder
    val_subdir: "images/valid"       # ← Adjust if using 'val'
  
  private:
    root: "./raw_data/private_dataset"
    train_subdir: "images/train"
    val_subdir: "images/valid"
    test_subdir: "images/test"       # Optional

classes:
  0: "license_plate"  # ← Must match class IDs in your label files
```

### 4. Data Preparation

```bash
# Step 1: Merge and preprocess datasets
python data_engine/prepare.py

# Step 2: Generate model-specific format files
python data_engine/converter.py
```

This creates:
- `processed_data/` directory with unified dataset
- `configs/final_data.yaml` for YOLO models (uses 0-indexed class IDs)
- `processed_data/annotations/*.json` for COCO-compatible models (uses 1-indexed category IDs)

**Note**: RT-DETRv2 requires COCO-format annotations with 1-indexed category IDs (standard COCO format). The converter automatically handles this conversion from YOLO's 0-indexed class IDs.

### 5. Training

```bash
# Train YOLOv11
python train.py --model yolov11 --config configs/base_config.yaml

# Train YOLOv10
python train.py --model yolov10 --config configs/base_config.yaml

# Train RT-DETRv2
python train.py --model rtdetrv2 --config configs/base_config.yaml
```

Training outputs are saved to `runs/{model}_run/`

### 6. Inference

```bash
# YOLO inference on single image
python infer.py --model yolo --weights runs/yolov11_run/weights/best.pt --source test.jpg

# RT-DETRv2 inference (auto-loads config from base_config.yaml)
python infer.py --model rtdetrv2 --weights runs/rtdetrv2_run/best.pth --source test.jpg

# Results saved to runs/inference/
```

## 📊 Supported Models

### YOLOv11
- **Framework**: Ultralytics
- **Variants**: nano, small, medium, large, extra-large
- **Format**: YOLO (normalized center-format)
- **Hyperparameters**: Configured in `base_config.yaml`
- **Best For**: Real-time inference with excellent accuracy

### YOLOv10
- **Framework**: Ultralytics
- **Variants**: nano, small, medium, balanced, large, extra-large
- **Format**: YOLO (normalized center-format)
- **Hyperparameters**: Configured in `base_config.yaml`
- **Best For**: End-to-end detection without NMS

### RT-DETRv2
- **Framework**: Official PyTorch
- **Format**: COCO JSON
- **Hyperparameters**: Uses author's config file (carefully tuned defaults)
- **Best For**: Transformer-based detection with high accuracy
- **Note**: To modify epochs/batch size, edit the RT-DETR config file directly

## 🎨 Data Augmentation

The system includes built-in augmentation pipeline with:
- Standardized resizing to target dimensions
- Horizontal flipping (50% probability)
- Random brightness/contrast adjustments
- Motion blur simulation
- Geometric transformations (shift, scale, rotate)

Configure augmentation in `base_config.yaml`:
```yaml
data_engine:
  augmentation:
    enable: true
    prob: 0.5
    params:
      horizontal_flip: true
      brightness_contrast: true
      blur_limit: 7
```

## 📝 Configuration Guide

### Dataset Configuration

```yaml
datasets:
  dataset_name:
    root: "/path/to/dataset"
    train_subdir: "images/train"
    val_subdir: "images/val"
    test_subdir: "images/test"  # Optional
```

### Model Hyperparameters

```yaml
models:
  # YOLOv11 and YOLOv10 - Hyperparameters are set here
  yolov11:
    model_name: "yolo11s.pt"
    epochs: 100      # Directly controls training
    batch: 16        # Directly controls batch size
  yolov10:
    model_name: "yolov10s.pt"
    epochs: 100      # Directly controls training
    batch: 16        # Directly controls batch size
  
  # RT-DETRv2 - Hyperparameters come from its own config file
  rtdetrv2:
    config_file: "models/rtdetr_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml"
    # To modify RT-DETR hyperparameters, edit the config_file directly
```

### Class Names

```yaml
classes:
  0: "license"
```

## 📈 Training Output

Training produces:
- **Weights**: `runs/{model}_run/weights/best.pt` (best model)
- **Metrics**: `runs/{model}_run/results.csv` (training curves)
- **Visualizations**: `runs/{model}_run/val/` (validation predictions)
