"""
YOLOv26 Trainer Module

Implements YOLOv26 training using the Ultralytics framework.
YOLOv26s is one of the latest YOLO variants, optimized for edge deployment
while retaining strong accuracy.

Author: ALPR Thesis Project
"""

import os
from ultralytics import YOLO
from .base_trainer import BaseTrainer


class YOLOv26Trainer(BaseTrainer):
    """
    YOLOv26 model trainer using Ultralytics framework.

    This trainer handles YOLOv26-specific model initialization and training
    using the standardized YOLO-format dataset configuration.
    """

    def train(self):
        """
        Execute YOLOv26 training pipeline.

        Loads the specified YOLOv26 variant and trains it on the processed
        dataset using hyperparameters from the configuration file.
        """
        print("--- Starting YOLOv26 Training ---")

        model_cfg = self.cfg["models"]["yolov26"]

        # Load pre-trained YOLOv26 model (e.g., 'yolo26s.pt')
        model_name = model_cfg["model_name"]
        model = YOLO(model_name)

        # Configure training output directory
        save_dir = os.path.join(self.cfg["output_dir"], "yolov26_run")

        # Execute training with Ultralytics trainer
        results = model.train(
            data=self.yolo_yaml,
            epochs=model_cfg["epochs"],
            batch=model_cfg["batch"],
            imgsz=self.cfg["data_engine"]["img_size"],
            save_dir=save_dir,
            workers=4,
            exist_ok=True,
            device=0,
            patience=model_cfg["patience"],
            **self.ultralytics_augmentation_kwargs(),
            **self.ultralytics_optimizer_lr_kwargs(model_cfg),
        )

        print(f"YOLOv26 Training Finished. Weights saved at: {save_dir}/weights/best.pt")

