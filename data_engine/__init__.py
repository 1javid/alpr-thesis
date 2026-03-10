"""
Data Engine Package - ALPR Object Detection System

This package handles all data processing operations including:
- Multi-dataset merging and preprocessing
- YOLO dataset configuration generation (Ultralytics)

Modules:
    prepare: Dataset merging and resize-only preprocessing
    converter: YOLO YAML generation for Ultralytics/RF-DETR
    augmentor: Resize-only preprocessing helper (legacy name)

Typical Workflow:
    1. Run prepare.py to merge and resize datasets
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
