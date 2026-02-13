"""
YOLOv10 Trainer Module

Implements YOLOv10 training using the Ultralytics framework.
YOLOv10 introduces end-to-end object detection without NMS (Non-Maximum Suppression),
providing faster inference while maintaining competitive accuracy.

Author: ALPR Thesis Project
"""

import os
from ultralytics import YOLO
from .base_trainer import BaseTrainer

class YOLOv10Trainer(BaseTrainer):
    """
    YOLOv10 model trainer using Ultralytics framework.
    
    This trainer handles YOLOv10-specific model initialization and training.
    YOLOv10 uses the same data format as other Ultralytics YOLO models.
    
    Supported YOLOv10 variants:
        - yolov10n.pt (nano)
        - yolov10s.pt (small)
        - yolov10m.pt (medium)
        - yolov10b.pt (balanced)
        - yolov10l.pt (large)
        - yolov10x.pt (extra-large)
    """
    
    def train(self):
        """
        Execute YOLOv10 training pipeline.
        
        Loads the specified YOLOv10 variant and trains it on the processed
        dataset using hyperparameters from the configuration file.
        
        Training outputs:
            - Trained weights: {output_dir}/yolov10_run/weights/best.pt
            - Training metrics: {output_dir}/yolov10_run/results.csv
            - Validation results: {output_dir}/yolov10_run/val/
        
        Configuration keys used:
            - models.yolov10.model_name: Pre-trained model variant (e.g., 'yolov10s.pt')
            - models.yolov10.epochs: Number of training epochs
            - models.yolov10.batch: Batch size for training
            - data_engine.img_size: Input image size (square)
            - output_dir: Base directory for training outputs
        """
        print("--- Starting YOLOv10 Training ---")

        # Load pre-trained YOLOv10 model
        model_name = self.cfg['models']['yolov10']['model_name']  # e.g., 'yolov10s.pt'
        model = YOLO(model_name)

        # Configure training output directory
        save_dir = os.path.join(self.cfg['output_dir'], 'yolov10_run')
        
        # Execute training with Ultralytics trainer
        # Uses same YOLO-format data configuration as other Ultralytics models
        results = model.train(
            data=self.yolo_yaml,                                    # Path to data YAML
            epochs=self.cfg['models']['yolov10']['epochs'],         # Training epochs
            batch=self.cfg['models']['yolov10']['batch'],           # Batch size
            imgsz=self.cfg['data_engine']['img_size'],              # Input image size
            save_dir=save_dir,                                       # Output directory
            workers=4,                                               # DataLoader workers
            exist_ok=True,                                           # Overwrite existing runs
            device=0                                                 # GPU device (0=first GPU, 'cpu' for CPU, [0,1,2,3] for multi-GPU)
        )
        
        print(f"YOLOv10 Training Finished. Weights saved at: {save_dir}/weights/best.pt")
