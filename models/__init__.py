"""
Models Package - ALPR Object Detection System

This package contains trainer implementations for multiple object detection
architectures including YOLOv11, YOLOv26, and RF-DETR-S.

Modules:
    base_trainer: Abstract base class for all trainers
    yolov11_trainer: YOLOv11 training implementation
    yolov26_trainer: YOLOv26 training implementation
    rfdetr_trainer: RF-DETR-S training implementation

Usage:
    from models.yolov11_trainer import YOLOv11Trainer
    from models.yolov26_trainer import YOLOv26Trainer
    from models.rfdetr_trainer import RFDETRSTrainer

Author: ALPR Thesis Project
"""

from .base_trainer import BaseTrainer
from .yolov11_trainer import YOLOv11Trainer
from .yolov26_trainer import YOLOv26Trainer
from .rfdetr_trainer import RFDETRSTrainer

__all__ = [
    'BaseTrainer',
    'YOLOv11Trainer',
    'YOLOv26Trainer',
    'RFDETRSTrainer',
]
