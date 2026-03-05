# ALPR Object Detection System

Multi-model object detection framework for license plate recognition research. Trains and compares YOLOv11, YOLOv10, and RT-DETRv2 using Ultralytics.

## Features

- Train and compare three models: YOLOv11-L, YOLOv10-L, RT-DETRv2-L
- All models use Ultralytics framework for consistency
- Merge multiple datasets automatically
- Data augmentation via Ultralytics (training-time)
- Single config file for everything
- Simple training and inference scripts

## Project Structure

```
alpr-thesis/
├── configs/
│   ├── base_config.yaml      # Master configuration file
│   └── final_data.yaml        # Auto-generated YOLO data config
├── data_engine/
│   ├── prepare.py             # Dataset merging and resize-only preprocessing
│   ├── converter.py           # Format conversion (YOLO ↔ COCO)
│   └── augmentor.py           # Resize-only preprocessing (legacy name)
├── models/
│   ├── base_trainer.py        # Abstract trainer base class
│   ├── yolov11_trainer.py     # YOLOv11 training implementation
│   ├── yolov10_trainer.py     # YOLOv10 training implementation
│   └── rtdetr_trainer.py      # RT-DETRv2 training implementation
├── train.py                   # Training entry point
├── infer.py                   # Inference entry point
└── requirements.txt           # Project dependencies
```

## Dataset Structure

Your datasets need to follow this structure:

### Directory Layout

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

### Label Format

YOLO format, one line per object:
```
class_id x_center y_center width height
```

Values are normalized (0-1). Example:
```
0 0.5 0.5 0.3 0.2
0 0.7 0.3 0.15 0.1
```

Update `configs/base_config.yaml` to match your folders:

```yaml
datasets:
  public:
    root: "./raw_data/public_dataset"
    train_subdir: "images/train"
    val_subdir: "images/valid"
  
  private:
    root: "./raw_data/private_dataset"
    train_subdir: "images/train"
    val_subdir: "images/valid"
    test_subdir: "images/test"  # optional
```

**Important:**
- Labels must mirror image directory structure
- Label filename matches image filename (`img.jpg` → `img.txt`)
- Both datasets need the same folder structure

## Installation

### Linux/macOS

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

git clone <repository-url>
cd alpr-thesis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
git clone <repository-url>
cd alpr-thesis

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Setup

### 1. Prepare Data

Put your datasets in `raw_data/`:
```bash
mkdir -p raw_data/public_dataset raw_data/private_dataset
```

### 2. Configure

Edit `configs/base_config.yaml`:
```yaml
datasets:
  public:
    root: "./raw_data/public_dataset"
    train_subdir: "images/train"
    val_subdir: "images/valid"

classes:
  0: "license_plate"
```

### 3. Process Data

```bash
python data_engine/prepare.py
python data_engine/converter.py
```

## Usage

### Training

```bash
python train.py --model yolov11 --config configs/base_config.yaml
python train.py --model yolov10 --config configs/base_config.yaml
python train.py --model rtdetrv2 --config configs/base_config.yaml
```

Weights saved to `runs/{model}_run/weights/best.pt`

### Inference

```bash
# Single image
python infer.py --model yolov11 --weights runs/yolov11_run/weights/best.pt --source test.jpg

# Directory
python infer.py --model yolov10 --weights runs/yolov10_run/weights/best.pt --source test_images/

# RT-DETRv2
python infer.py --model rtdetrv2 --weights runs/rtdetrv2_run/weights/best.pt --source test.jpg
```

Results saved to `runs/inference/{model}/`

## Models

All models use Large variants via Ultralytics:

- **YOLOv11-L** (`yolo11l.pt`) - Latest YOLO, best accuracy
- **YOLOv10-L** (`yolov10l.pt`) - End-to-end, no NMS
- **RT-DETRv2-L** (`rtdetr-l.pt`) - Transformer-based

## Configuration

Edit `configs/base_config.yaml`:

```yaml
# Models
models:
  yolov11:
    model_name: "yolo11l.pt"
    epochs: 100
    batch: 16
  yolov10:
    model_name: "yolov10l.pt"
    epochs: 100
    batch: 16
  rtdetrv2:
    model_name: "rtdetr-l.pt"
    epochs: 100
    batch: 16

# Classes
classes:
  0: "license"

# Augmentation
data_engine:
  img_size: 640
  augmentation:
    enable: true
    prob: 0.5
```

**Batch sizes for Large models:**
- 16GB GPU: 8-16
- 24GB GPU: 16-32

## Output

Training saves to `runs/{model}_run/`:
- `weights/best.pt` - Best model weights
- `results.csv` - Training metrics
- `val/` - Validation visualizations
