"""
Data preprocessing helper (legacy `DataAugmentor` name kept).

This project previously generated *offline* augmented images using Albumentations.
Augmentation has been migrated to Ultralytics' built-in training pipeline, so this
module now provides **resize-only preprocessing** used by `data_engine/prepare.py`.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2


def load_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Load YOLO-format labels from disk.

    Returns:
        (bboxes, class_ids) where bboxes are [xc, yc, w, h] normalized floats.
    """
    bboxes: List[List[float]] = []
    class_ids: List[int] = []

    if not label_path or not os.path.exists(label_path):
        return bboxes, class_ids

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
            except ValueError:
                continue
            class_ids.append(cls_id)
            bboxes.append(bbox)

    return bboxes, class_ids


class DataAugmentor:
    """
    Resize-only preprocessing for object detection datasets.

    Notes:
        - Images are resized to a fixed square `img_size`.
        - YOLO labels are normalized; pure resize keeps bbox coordinates unchanged.
        - Offline augmentation is intentionally not supported anymore.
    """

    def __init__(self, cfg: dict):
        self.target_size = int(cfg.get("img_size", 640))

    def process(self, img_path: str, label_path: str):
        raise NotImplementedError(
            "Offline augmentation has been removed. "
            "Use Ultralytics training-time augmentation instead."
        )

    def process_resize_only(self, img_path: str, label_path: str):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            return None, None, None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(
            image_rgb,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        bboxes, class_ids = load_yolo_labels(label_path)
        return resized_rgb, bboxes, class_ids
