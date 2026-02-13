# RT-DETRv2 Integration Guide

This document describes all modifications and integration steps for RT-DETRv2 in the ALPR thesis project.

## Overview

RT-DETRv2 (Real-Time DEtection TRansformer v2) is integrated as one of three object detection models for license plate detection, alongside YOLOv11 and YOLOv10.

**Original Repository:** https://github.com/lyuwenyu/RT-DETR  
**Integration Location:** `models/rtdetr_pytorch/`

---

## External Integration (Project-Level)

### 1. Project Structure

```
alpr-thesis/
├── models/
│   ├── rtdetr_trainer.py          # Custom trainer wrapper
│   ├── rtdetr_pytorch/            # Official RT-DETR repository
│   │   ├── src/                   # RT-DETR source code
│   │   ├── configs/               # Model configurations
│   │   └── INTEGRATION.md         # This file
├── configs/
│   └── base_config.yaml           # Project config (includes rtdetrv2 settings)
├── train.py                       # Main training script
├── infer.py                       # Main inference script
└── processed_data/                # COCO format dataset
```

### 2. Custom Trainer (`models/rtdetr_trainer.py`)

**Purpose:** Wraps the official RT-DETR training API to work with the unified project interface.

**Key Features:**
- Dynamically adds `models/rtdetr_pytorch` to Python path
- Loads RT-DETR's YAMLConfig system
- Injects custom dataset paths into the official config
- Overrides training parameters (epochs, batch size, output directory)
- Handles single-class detection (license plates)

**Code Highlights:**

```python
# Path injection
sys.path.insert(0, os.path.join(project_root, 'models', 'rtdetr_pytorch'))

# Import official modules
from src.core import YAMLConfig
from src.solver import TASKS

# Override data paths
yaml_cfg['train_dataloader']['dataset']['img_folder'] = 'processed_data/images/train'
yaml_cfg['train_dataloader']['dataset']['ann_file'] = 'processed_data/annotations/instances_train.json'

# Override training settings
yaml_cfg['epoches'] = epochs
yaml_cfg['output_dir'] = 'runs/rtdetrv2_run'
yaml_cfg['num_classes'] = 1
yaml_cfg['remap_mscoco_category'] = False  # CRITICAL: Disable COCO remapping
```

### 3. Main Training Script Integration (`train.py`)

```python
# Conditional import to avoid crashes if RT-DETR is missing
if args.model == 'rtdetrv2':
    from models.rtdetr_trainer import RTDETRv2Trainer
    trainer = RTDETRv2Trainer(cfg)
```

### 4. Inference Script Integration (`infer.py`)

**Model Loading:**
```python
elif self.args.model == 'rtdetrv2':
    from src.core import YAMLConfig
    cfg = YAMLConfig(self.args.model_config, resume=self.args.weights)
    model = cfg.model
    checkpoint = torch.load(self.args.weights, map_location='cuda:0')
    model.load_state_dict(checkpoint['ema']['module'])
    model.cuda().eval()
```

**Postprocessing:**
- Converts relative (cx, cy, w, h) boxes to absolute (x1, y1, x2, y2)
- Applies confidence threshold (0.5)
- Draws predictions on images

---

## Internal Modifications (RT-DETR Config Files)

### 1. Configuration File Structure

RT-DETR uses YAML configs with `__include__` directives for modular composition.

**Base Config Used:**
```
models/rtdetr_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml
```

### 2. Key Modifications in Config Files

#### **Category Remapping Disabled**

**File:** Config YAML (set dynamically in `rtdetr_trainer.py`)

```yaml
# ORIGINAL (COCO mode)
remap_mscoco_category: True  # Maps COCO IDs 1-90

# MODIFIED (Custom dataset mode)
remap_mscoco_category: False  # Uses raw category IDs (0-based)
```

**Why:** Our annotations use `category_id: 0` for license plates. COCO remapping would break this.

#### **Number of Classes**

```yaml
# ORIGINAL
num_classes: 80  # COCO classes

# MODIFIED
num_classes: 1   # License plate only
```

**Set dynamically in:** `rtdetr_trainer.py` line 68

#### **Dataset Paths**

```yaml
# ORIGINAL
train_dataloader:
  dataset:
    img_folder: /data/coco/train2017
    ann_file: /data/coco/annotations/instances_train2017.json

# MODIFIED (set dynamically)
train_dataloader:
  dataset:
    img_folder: processed_data/images/train
    ann_file: processed_data/annotations/instances_train.json
```

**Set dynamically in:** `rtdetr_trainer.py` lines 46-55

#### **Training Parameters**

```yaml
# ORIGINAL
epoches: 120
total_batch_size: 32
output_dir: ./output/rtdetrv2_r18vd

# MODIFIED (set dynamically from base_config.yaml)
epoches: 2                    # From models.rtdetrv2.epochs
total_batch_size: 16          # From models.rtdetrv2.batch
output_dir: runs/rtdetrv2_run # Project-specific
```

**Set dynamically in:** `rtdetr_trainer.py` lines 50, 55, 59-65

### 3. Include Hierarchy

The config system uses nested includes:

```
rtdetrv2_r18vd_sp3_120e_coco.yml
├── __include__: ['include/rtdetrv2_r50vd.yml', 'include/dataloader.yml']
│   └── include/rtdetrv2_r50vd.yml
│       └── Defines model architecture
│   └── include/dataloader.yml
│       └── Defines data loading pipeline
└── Overrides for specific variant (r18vd, sp3)
```

**No modifications required** - include paths resolve correctly when config is loaded from the official repo directory.

---

## Dependencies

### Added to `requirements.txt`

```txt
# Model 3: RT-DETRv2 (Official PyTorch Dependencies)
submitit
onnx
onnxruntime
tensorboard
faster-coco-eval>=1.6.6
pycocotools
```

### System Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

---

## Usage

### Training

```bash
# Basic training
python train.py --model rtdetrv2 --config configs/base_config.yaml

# Custom epochs/batch size (edit base_config.yaml first)
# configs/base_config.yaml:
#   models:
#     rtdetrv2:
#       epochs: 50
#       batch: 8
```

**Output Location:** `runs/rtdetrv2_run/`
- `best.pth` - Best model checkpoint
- `last.pth` - Latest model checkpoint
- Logs and metrics

### Inference

```bash
# Single image
python infer.py \
  --model rtdetrv2 \
  --weights runs/rtdetrv2_run/best.pth \
  --model_config models/rtdetr_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml \
  --source path/to/image.jpg

# Directory of images
python infer.py \
  --model rtdetrv2 \
  --weights runs/rtdetrv2_run/best.pth \
  --model_config models/rtdetr_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml \
  --source path/to/images/
```

**Output Location:** `runs/inference/rtdetrv2/`

---

## Data Format

### Input Format (COCO JSON)

RT-DETRv2 expects COCO format annotations:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "train_0001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 12345.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "license"}
  ]
}
```

**Generated automatically** by `data_engine/converter.py` from YOLO format.

### Dataset Directory Structure

```
processed_data/
├── images/
│   ├── train/
│   │   ├── train_0001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

---

## Model Variants

RT-DETRv2 supports multiple backbones and training strategies:

| Config File | Backbone | Training Strategy | Parameters |
|-------------|----------|-------------------|------------|
| `rtdetrv2_r18vd_sp3_120e_coco.yml` | ResNet-18 | Sparse 3 | ~20M |
| `rtdetrv2_r34vd_120e_coco.yml` | ResNet-34 | Standard | ~31M |
| `rtdetrv2_r50vd_6x_coco.yml` | ResNet-50 | Standard | ~42M |
| `rtdetrv2_hgnetv2_l_6x_coco.yml` | HGNetV2-L | Standard | ~32M |

**To change variant:** Update `config_file` in `configs/base_config.yaml`:

```yaml
models:
  rtdetrv2:
    config_file: "models/rtdetr_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    epochs: 50
    batch: 8
```

---

## Troubleshooting

### Issue: `ImportError: cannot import name 'YAMLConfig'`

**Cause:** RT-DETR repository not present or path issue.

**Solution:**
```bash
cd models/
git clone https://github.com/lyuwenyu/RT-DETR.git rtdetr_pytorch
```

### Issue: Training crashes with COCO category errors

**Cause:** `remap_mscoco_category` is True.

**Solution:** Ensure `rtdetr_trainer.py` line 70 sets:
```python
yaml_cfg['remap_mscoco_category'] = False
```

### Issue: No detections during inference

**Cause:** Wrong checkpoint key or model not trained.

**Solution:**
1. Check checkpoint has `ema.module` state dict
2. Verify training completed successfully
3. Lower confidence threshold in `infer.py` line 110

### Issue: GPU out of memory

**Solution:**
- Reduce batch size in `base_config.yaml`
- Use smaller backbone (r18vd instead of r50vd)
- Reduce image size (edit `img_size` in `base_config.yaml`)

---

## Key Differences from Official RT-DETR

| Aspect | Official RT-DETR | This Integration |
|--------|------------------|------------------|
| **Data format** | Expects COCO dataset | Uses `processed_data/` with custom paths |
| **Category IDs** | COCO 1-90 remapping | Single class (0), remapping disabled |
| **Config loading** | Standalone CLI tool | Wrapped in `rtdetr_trainer.py` |
| **Training entry** | `tools/train.py` | `train.py --model rtdetrv2` |
| **Output directory** | `./output/` | `runs/rtdetrv2_run/` |
| **Inference** | Separate tool | Unified `infer.py` |

---

## References

- **Original Paper:** [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- **Official Repo:** https://github.com/lyuwenyu/RT-DETR
- **Model Weights:** Pretrained on COCO (downloaded automatically on first training)

---

## Summary of All Modifications

### Files Created
1. `models/rtdetr_trainer.py` - Custom trainer wrapper
2. `models/rtdetr_pytorch/INTEGRATION.md` - This file

### Files Modified
1. `train.py` - Added rtdetrv2 model option
2. `infer.py` - Added rtdetrv2 inference logic
3. `configs/base_config.yaml` - Added rtdetrv2 config section
4. `requirements.txt` - Added RT-DETR dependencies

### Configuration Overrides (Dynamic)
1. `remap_mscoco_category: False` - Disable COCO ID remapping
2. `num_classes: 1` - Single license plate class
3. Dataset paths → `processed_data/`
4. Output directory → `runs/rtdetrv2_run/`
5. Training parameters → From `base_config.yaml`

### No Direct File Modifications in `rtdetr_pytorch/`
All modifications are applied **dynamically at runtime** through the wrapper, preserving the original RT-DETR repository structure.
