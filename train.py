"""
Train Module - ALPR Object Detection System

This module provides a unified training interface for multiple object detection models.
Supports YOLOv11, YOLOv26 (Ultralytics), and RF-DETR-S (Roboflow) architectures
with configuration-driven training.

Usage:
    python train.py --model yolov11 --config configs/base_config.yaml
    python train.py --model yolov26 --config configs/base_config.yaml
    python train.py --model rfdetr --config configs/base_config.yaml

Author: ALPR Thesis Project
"""

import argparse
import os
import sys

import yaml

# Ensure current directory is in Python path for module imports
sys.path.append(os.getcwd())

from models.yolov11_trainer import YOLOv11Trainer
from models.yolov26_trainer import YOLOv26Trainer
from models.rfdetr_trainer import RFDETRSTrainer
from utils.seed_utils import get_seed_from_config, set_global_seed

def main():
    """
    Main training entry point.
    
    Parses command-line arguments, loads configuration, and initializes
    the appropriate model trainer based on user selection.
    
    Command-line Arguments:
        --model: Model architecture to train (yolov11, yolov26, rfdetr)
        --config: Path to YAML configuration file (default: configs/base_config.yaml)
    
    Raises:
        FileNotFoundError: If configuration file does not exist
        ValueError: If model type is not supported
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Modular Object Detection Trainer")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolov11', 'yolov26', 'rfdetr'],
        help="Model architecture to train: 'yolov11', 'yolov26', or 'rfdetr'"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/base_config.yaml', 
        help="Path to master configuration file"
    )
    args = parser.parse_args()

    # Load master configuration from YAML file
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply global experiment seed for reproducibility
    seed = get_seed_from_config(cfg)
    set_global_seed(seed)
    print(f"Initializing training for model: {args.model} (seed={seed})")

    # Initialize appropriate trainer based on model selection
    if args.model == 'yolov11':
        trainer = YOLOv11Trainer(cfg)
    elif args.model == 'yolov26':
        trainer = YOLOv26Trainer(cfg)
    elif args.model == 'rfdetr':
        trainer = RFDETRSTrainer(cfg)

    # Execute training loop
    trainer.train()

if __name__ == "__main__":
    main()