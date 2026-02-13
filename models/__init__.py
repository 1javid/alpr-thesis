"""
Models Package - ALPR Object Detection System

This package contains trainer implementations for multiple object detection
architectures including YOLOv11, YOLOv10, and RT-DETRv2.

Modules:
    base_trainer: Abstract base class for all trainers
    yolov11_trainer: YOLOv11 training implementation
    yolov10_trainer: YOLOv10 training implementation
    rtdetr_trainer: RT-DETRv2 training implementation (optional)

Usage:
    from models.yolov11_trainer import YOLOv11Trainer
    from models.yolov10_trainer import YOLOv10Trainer

Author: ALPR Thesis Project
"""

from .base_trainer import BaseTrainer
from .yolov11_trainer import YOLOv11Trainer
from .yolov10_trainer import YOLOv10Trainer

__all__ = [
    'BaseTrainer',
    'YOLOv11Trainer',
    'YOLOv10Trainer',
]
