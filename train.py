"""
Train Module - ALPR Object Detection System

This module provides a unified training interface for multiple object detection models.
It supports YOLOv11, YOLOv10, and RT-DETRv2 architectures with configuration-driven training.

Usage:
    python train.py --model yolov11 --config configs/base_config.yaml
    python train.py --model yolov10 --config configs/base_config.yaml
    python train.py --model rtdetrv2 --config configs/base_config.yaml

Author: ALPR Thesis Project
"""

import argparse
import yaml
import os
import sys

# Ensure current directory is in Python path for module imports
sys.path.append(os.getcwd())

from models.yolov11_trainer import YOLOv11Trainer
from models.yolov10_trainer import YOLOv10Trainer

def main():
    """
    Main training entry point.
    
    Parses command-line arguments, loads configuration, and initializes
    the appropriate model trainer based on user selection.
    
    Command-line Arguments:
        --model: Model architecture to train (yolov11, yolov10, rtdetrv2)
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
        choices=['yolov11', 'yolov10', 'rtdetrv2'],
        help="Model architecture to train: 'yolov11', 'yolov10', or 'rtdetrv2' (Official PyTorch)"
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

    print(f"Initializing training for model: {args.model}")

    # Initialize appropriate trainer based on model selection
    if args.model == 'yolov11':
        trainer = YOLOv11Trainer(cfg)
        
    elif args.model == 'yolov10':
        trainer = YOLOv10Trainer(cfg)

    elif args.model == 'rtdetrv2':
        # Lazy import for RT-DETRv2 to avoid unnecessary dependencies
        from models.rtdetr_trainer import RTDETRv2Trainer
        trainer = RTDETRv2Trainer(cfg)

    # Execute training loop
    trainer.train()

if __name__ == "__main__":
    main()