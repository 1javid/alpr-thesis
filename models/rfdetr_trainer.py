"""
RF-DETR-S Trainer Module

Implements RF-DETR-S training using the `rfdetr` library.
RF-DETR is a transformer-based real-time detector developed by Roboflow.

This trainer expects a YOLO or COCO dataset layout under `processed_data/`,
and uses the RF-DETR automatic dataset format detection when `dataset_dir`
is pointed at the dataset root.

Author: ALPR Thesis Project
"""

import os
from .base_trainer import BaseTrainer


class RFDETRSTrainer(BaseTrainer):
    """
    RF-DETR-S model trainer using the `rfdetr` package.

    Configuration block used:
        models.rfdetr:
            - model_size: model size string (e.g. 's')
            - epochs
            - batch
            - patience
            - lr
    """

    def train(self):
        """
        Execute RF-DETR-S training pipeline.

        Uses the processed dataset directory as the RF-DETR `dataset_dir`.
        The dataset must follow either YOLO or COCO format as supported
        by the `rfdetr` library.
        """
        print("--- Starting RF-DETR-S Training ---")

        try:
            from rfdetr import RFDETRBase  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'rfdetr' package is required for RF-DETR-S training. "
                "Install it with `pip install rfdetr`."
            ) from e

        model_cfg = self.cfg["models"]["rfdetr"]

        # dataset_dir: we point RF-DETR at the processed data root.
        # RF-DETR will auto-detect YOLO vs COCO format.
        dataset_dir = self.data_root

        # Output directory for RF-DETR runs
        save_dir = os.path.join(self.cfg["output_dir"], "rfdetr_run")
        os.makedirs(save_dir, exist_ok=True)

        model = RFDETRBase(model_size=model_cfg.get("model_size", "s"))

        # RF-DETR has its own training API; we pass common hyperparameters.
        model.train(
            dataset_dir=dataset_dir,
            epochs=model_cfg["epochs"],
            batch_size=model_cfg["batch"],
            lr=model_cfg["lr"],
            output_dir=save_dir,
        )

        print(f"RF-DETR-S Training Finished. Weights saved under: {save_dir}")

