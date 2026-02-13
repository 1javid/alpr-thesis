"""
Data Engine Package - ALPR Object Detection System

This package handles all data processing operations including:
- Multi-dataset merging and preprocessing
- Data augmentation with bounding box support
- Format conversion (YOLO ↔ COCO)

Modules:
    prepare: Dataset merging, preprocessing, and augmentation
    converter: Format conversion between YOLO and COCO formats
    augmentor: Image augmentation pipeline using Albumentations

Typical Workflow:
    1. Run prepare.py to merge and augment datasets
    2. Run converter.py to generate model-specific format files
    3. Use generated files for model training

Usage:
    from data_engine.augmentor import DataAugmentor
    from data_engine.converter import DataConverter

Author: ALPR Thesis Project
"""

from .augmentor import DataAugmentor
from .converter import DataConverter

__all__ = [
    'DataAugmentor',
    'DataConverter',
]
