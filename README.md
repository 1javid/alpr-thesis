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

### 2. Configuration

Edit `configs/base_config.yaml` to specify:
- Dataset paths and structure
- Class names and IDs
- Image size and augmentation parameters
- Model-specific hyperparameters

### 3. Data Preparation

```bash
# Step 1: Merge and preprocess datasets
python data_engine/prepare.py

# Step 2: Generate model-specific format files
python data_engine/converter.py
```

This creates:
- `processed_data/` directory with unified dataset
- `configs/final_data.yaml` for YOLO models
- `processed_data/annotations/*.json` for COCO-compatible models

### 4. Training

```bash
# Train YOLOv11
python train.py --model yolov11 --config configs/base_config.yaml

# Train YOLOv10
python train.py --model yolov10 --config configs/base_config.yaml

# Train RT-DETRv2
python train.py --model rtdetrv2 --config configs/base_config.yaml
```

Training outputs are saved to `runs/{model}_run/`

### 5. Inference

```bash
# YOLO inference on single image
python infer.py --model yolo --weights runs/yolov11_run/weights/best.pt --source test.jpg

# RT-DETRv2 inference on directory
python infer.py --model rtdetrv2 --weights checkpoint.pth --model_config config.yaml --source test_images/

# Results saved to runs/inference/
```

## 📊 Supported Models

### YOLOv11
- **Framework**: Ultralytics
- **Variants**: nano, small, medium, large, extra-large
- **Format**: YOLO (normalized center-format)
- **Best For**: Real-time inference with excellent accuracy

### YOLOv10
- **Framework**: Ultralytics
- **Variants**: nano, small, medium, balanced, large, extra-large
- **Format**: YOLO (normalized center-format)
- **Best For**: End-to-end detection without NMS

### RT-DETRv2
- **Framework**: Official PyTorch
- **Format**: COCO JSON
- **Best For**: Transformer-based detection with high accuracy

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
  yolov11:
    model_name: "yolo11s.pt"
    epochs: 100
    batch: 16
  yolov10:
    model_name: "yolov10s.pt"
    epochs: 100
    batch: 16
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
