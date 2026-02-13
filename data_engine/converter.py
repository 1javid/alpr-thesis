"""
Data Converter Module

Converts processed YOLO-format datasets into multiple model-specific formats.
Generates configuration files for Ultralytics YOLO models and COCO JSON annotations
for RT-DETR and other COCO-compatible frameworks.

Key Features:
    - YOLO YAML generation for Ultralytics models (YOLOv11, YOLOv10)
    - COCO JSON generation for RT-DETR and MMDetection-compatible models
    - Automatic format conversion from normalized YOLO to pixel-based COCO
    - Support for train/val/test splits

Output Formats:
    1. YOLO YAML: configs/final_data.yaml
    2. COCO JSON: {target_path}/annotations/instances_{split}.json

Author: ALPR Thesis Project
"""

import os
import yaml
import json
import cv2
from tqdm import tqdm
from PIL import Image

class DataConverter:
    """
    Multi-format dataset converter for object detection models.
    
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
        # Structure: {target_root}/images/{train,val,test}
        self.splits = {
            'train': 'images/train',
            'val': 'images/val',
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

    def convert_to_coco(self):
        """
        Convert YOLO-format data to COCO JSON format for RT-DETR and other models.
        
        Converts normalized YOLO bounding boxes to pixel-based COCO format,
        creating separate JSON annotation files for each dataset split.
        
        COCO Format Structure:
            {
                "images": [{"id", "file_name", "height", "width"}, ...],
                "annotations": [{"id", "image_id", "category_id", "bbox", "area", "iscrowd"}, ...],
                "categories": [{"id", "name"}, ...]
            }
        
        Bounding Box Conversion:
            YOLO (normalized): [x_center, y_center, width, height]
            COCO (pixel): [x_min, y_min, width, height]
        
        Output:
            {target_root}/annotations/instances_{split}.json for each split
        
        Note:
            Uses PIL for image size reading (faster than OpenCV).
            Handles malformed label files gracefully by skipping invalid entries.
        """
        # Build COCO categories from class mapping
        categories = [{"id": k, "name": v} for k, v in self.names.items()]
        
        print(f"[RT-DETR] Converting processed data to COCO format...")

        # Create annotations directory
        output_dir = os.path.join(self.target_root, 'annotations')
        os.makedirs(output_dir, exist_ok=True)

        # Process each split (train/val/test)
        for split, rel_path in self.splits.items():
            img_dir = os.path.join(self.target_root, rel_path)
            if not os.path.exists(img_dir):
                continue
            
            # Determine corresponding label directory
            lbl_dir = img_dir.replace('images', 'labels')
            
            # Initialize COCO data structures
            images = []
            annotations = []
            ann_id = 0
            
            img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
            
            # Process each image in the split
            for img_id, img_file in enumerate(tqdm(img_files, desc=f"Converting {split}")):
                img_path = os.path.join(img_dir, img_file)
                
                # Read image dimensions (PIL is faster than OpenCV for this)
                with Image.open(img_path) as img:
                    w, h = img.size

                # Add image metadata to COCO format
                images.append({
                    "id": img_id,
                    "file_name": img_file,
                    "height": h,
                    "width": w
                })
                
                # Read and convert YOLO labels to COCO annotations
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_file)
                
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            # Malformed line, skip
                            continue
                        
                        # Parse class ID (handle float-like strings, e.g., "0.0")
                        try:
                            cls_id = int(float(parts[0]))
                        except ValueError:
                            # Cannot parse class ID, skip this line
                            continue
                        
                        # Parse YOLO bbox (normalized coordinates)
                        x_c, y_c, w_bbox, h_bbox = map(float, parts[1:5])
                        
                        # Convert YOLO (normalized) to COCO (pixel) format
                        # YOLO: [x_center, y_center, width, height] (0-1)
                        # COCO: [x_min, y_min, width, height] (pixels)
                        x_min = (x_c - w_bbox/2) * w
                        y_min = (y_c - h_bbox/2) * h
                        width = w_bbox * w
                        height = h_bbox * h
                        
                        # Add annotation to COCO format
                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        ann_id += 1
            
            # Assemble complete COCO JSON structure
            coco_output = {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }
            
            # Save COCO JSON file
            json_name = f"instances_{split}.json"
            json_path = os.path.join(output_dir, json_name)
            with open(json_path, 'w') as f:
                json.dump(coco_output, f)
            
            print(f"  -> {split}: {len(images)} images, {len(annotations)} annotations")

        print(f"[RT-DETR] COCO annotations saved to: {output_dir}")

if __name__ == "__main__":
    """
    Command-line execution for data format conversion.
    
    Usage:
        python converter.py
    
    Generates:
        1. configs/final_data.yaml (YOLO format)
        2. {target_path}/annotations/instances_*.json (COCO format)
    """
    print("=== Data Format Converter ===\n")
    
    c = DataConverter()
    
    # Generate YOLO configuration for Ultralytics models
    c.generate_yolo_yaml()
    
    # Generate COCO JSON annotations for RT-DETR and compatible models
    c.convert_to_coco()
    
    print("\n=== Conversion Complete ===")
    print("Models can now be trained using the generated configuration files.")
