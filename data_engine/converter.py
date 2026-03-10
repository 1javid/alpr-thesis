"""
Data Converter Module

Generates:
  1. Ultralytics YOLO dataset configuration YAML (for YOLO11/YOLO26)
  2. RF-DETR-compatible COCO JSON annotations.

COCO JSONs are written as:
  processed_data/{train,val,test}/_annotations.coco.json
with `file_name` entries pointing to the existing YOLO image paths
(`images/{split}/image.jpg`), so we do not duplicate image files.

Author: ALPR Thesis Project
"""

import os
import yaml
import json
from tqdm import tqdm
from PIL import Image


class DataConverter:
    """
    Dataset converter for object detection models.
    
    This class handles conversion from the unified YOLO-format processed data
    into specific formats required by different model architectures.
    
    Attributes:
        cfg (dict): Master configuration dictionary
        target_root (str): Path to processed data directory
        names (dict): Class ID to class name mapping
        splits (dict): Dataset split name to relative path mapping
    """
    def __init__(self, base_config_path="configs/base_config.yaml"):
        """
        Initialize data converter with configuration.
        
        Args:
            base_config_path (str): Path to master configuration YAML file
        
        Loads:
            - Target data directory path
            - Class names mapping
            - Dataset split structure
        """
        with open(base_config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.target_root = self.cfg['data_engine']['target_path']
        self.names = self.cfg['classes']  # {0: "class_name", ...}

        # Define standardized directory structure created by prepare.py
        # Structure: {target_root}/images/{train,valid,test}
        # NOTE: RF-DETR expects the split name 'valid' (not 'val'), so we
        # map 'valid' -> 'images/val' here for COCO export.
        self.splits = {
            'train': 'images/train',
            'valid': 'images/val',
            'test': 'images/test',
        }

    def generate_yolo_yaml(self, save_name="final_data.yaml"):
        """
        Generate YAML configuration file for Ultralytics YOLO models.
        
        Creates a configuration file that Ultralytics YOLO models (YOLOv10, YOLOv11)
        use to locate training/validation/test data and understand class mappings.
        
        Args:
            save_name (str): Output filename for YAML config
        
        Returns:
            str: Full path to generated YAML file
        
        Output Format:
            path: /absolute/path/to/processed_data
            train: images/train
            val: images/val
            test: images/test  # Optional, if test split exists
            names:
              0: class_name_1
              1: class_name_2
              ...
        
        Note:
            The 'path' field uses absolute path for portability across
            different working directories.
        """
        # Get absolute path to processed data for portability
        abs_root = os.path.abspath(self.target_root)
        
        # Build YOLO configuration dictionary
        yolo_config = {
            'path': abs_root,
            'train': 'images/train',
            'val': 'images/val',
            'names': self.names
        }
        
        # Include test split if it exists
        if os.path.exists(os.path.join(abs_root, 'images/test')):
            yolo_config['test'] = 'images/test'

        # Save configuration to file
        save_path = os.path.join('configs', save_name)
        with open(save_path, 'w') as f:
            yaml.dump(yolo_config, f, sort_keys=False)
        
        print(f"[Ultralytics] YOLO config saved to: {save_path}")
        return save_path

    def convert_to_coco_for_rfdetr(self):
        """
        Convert YOLO-format data to COCO JSON format for RF-DETR.
        
        For each split (train/val/test) that exists under:
            {target_root}/images/{split}
        we create:
            {target_root}/{split}/_annotations.coco.json
        
        The generated COCO JSON has:
            images:     file_name set to 'images/{split}/{image_name}'
            annotations: bbox in COCO pixel format [x_min, y_min, w, h]
            categories:  1-indexed category IDs
        """
        categories = [{"id": k + 1, "name": v, "supercategory": "none"} for k, v in self.names.items()]

        print("[RF-DETR] Converting processed data to COCO format...")

        for split, rel_path in self.splits.items():
            img_dir = os.path.join(self.target_root, rel_path)
            if not os.path.exists(img_dir):
                continue

            lbl_dir = img_dir.replace("images", "labels")

            images = []
            annotations = []
            ann_id = 0

            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for img_id, img_file in enumerate(tqdm(img_files, desc=f"Converting {split}")):
                img_path = os.path.join(img_dir, img_file)

                with Image.open(img_path) as img:
                    w, h = img.size

                # RF-DETR resolves: dataset_dir/split/file_name.
                # Our images are at: processed_data/images/{rel_path}/img.
                # COCO JSON is at:   processed_data/{split}/_annotations.coco.json.
                # So file_name must go one level up: ../images/{split_dir}/img.
                file_name = os.path.join("..", rel_path, img_file)

                images.append(
                    {
                        "id": img_id,
                        "file_name": file_name,
                        "height": h,
                        "width": w,
                    }
                )

                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_file)

                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        try:
                            yolo_class_id = int(float(parts[0]))
                        except ValueError:
                            continue

                        coco_category_id = yolo_class_id + 1

                        x_c, y_c, w_bbox, h_bbox = map(float, parts[1:5])

                        x_min = (x_c - w_bbox / 2) * w
                        y_min = (y_c - h_bbox / 2) * h
                        width = w_bbox * w
                        height = h_bbox * h

                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": coco_category_id,
                                "bbox": [x_min, y_min, width, height],
                                "area": width * height,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

            coco_output = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }

            # Write to {target_root}/{split}/_annotations.coco.json
            split_dir = os.path.join(self.target_root, split)
            os.makedirs(split_dir, exist_ok=True)
            json_path = os.path.join(split_dir, "_annotations.coco.json")
            with open(json_path, "w") as f:
                json.dump(coco_output, f)

            print(f"  -> {split}: {len(images)} images, {len(annotations)} annotations")

        print("[RF-DETR] COCO annotations saved under processed_data/{train,val,test}/_annotations.coco.json")


if __name__ == "__main__":
    """
    Command-line execution for data format conversion.
    
    Usage:
        python converter.py
    
    Generates:
        1. configs/final_data.yaml (YOLO format)
        2. processed_data/{train,val,test}/_annotations.coco.json (RF-DETR COCO format)
    """
    print("=== Data Format Converter ===\n")
    
    c = DataConverter()
    
    # Generate YOLO configuration for Ultralytics models
    c.generate_yolo_yaml()

    # Generate COCO JSON annotations for RF-DETR
    c.convert_to_coco_for_rfdetr()
    
    print("\n=== Conversion Complete ===")
    print("Models can now be trained using the generated configuration files.")
