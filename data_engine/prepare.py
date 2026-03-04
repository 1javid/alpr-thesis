"""
Data Preparation Module

Merges and preprocesses multiple object detection datasets into a unified format.
Handles dataset merging, image resizing, augmentation, and standardized directory
structure creation for training, validation, and test splits.

Key Features:
    - Multi-dataset merging with namespace protection
    - Standardized resize to target dimensions
    - Training data augmentation (optional)
    - YOLO-format output (images + text labels)
    - Organized directory structure

Directory Structure Created:
    {target_path}/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/

Author: ALPR Thesis Project
"""

import os
import shutil

import cv2
import yaml
from tqdm import tqdm

from augmentor import DataAugmentor
from utils.seed_utils import get_seed_from_config, set_global_seed

def save_yolo_data(save_dir, split, filename, image, bboxes, class_ids):
    """
    Save image and corresponding YOLO-format labels to disk.
    
    Args:
        save_dir (str): Root directory for processed data
        split (str): Dataset split ('train', 'val', or 'test')
        filename (str): Original filename (will be converted to .jpg)
        image (np.ndarray): RGB image array to save
        bboxes (list): List of bounding boxes in YOLO format [xc, yc, w, h]
        class_ids (list): List of class IDs corresponding to bboxes
    
    Output Structure:
        {save_dir}/images/{split}/{filename}.jpg
        {save_dir}/labels/{split}/{filename}.txt
    
    Note:
        Images are converted from RGB to BGR before saving with OpenCV.
    """
    # Save image file
    # Convert RGB back to BGR for OpenCV compatibility
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_name = os.path.splitext(filename)[0] + ".jpg"
    img_path = os.path.join(save_dir, 'images', split, img_name)
    cv2.imwrite(img_path, img_bgr)

    # Save label file in YOLO format
    lbl_name = os.path.splitext(filename)[0] + ".txt"
    lbl_path = os.path.join(save_dir, 'labels', split, lbl_name)
    
    with open(lbl_path, 'w') as f:
        for cls, bbox in zip(class_ids, bboxes):
            # Write each object: class_id x_center y_center width height (all normalized)
            line = f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)

def main():
    """
    Main data preparation pipeline.
    
    Workflow:
        1. Load configuration from base_config.yaml
        2. Create standardized output directory structure
        3. Initialize augmentation pipeline
        4. Process each dataset:
           - Merge multiple datasets into unified structure
           - Apply resize to all images
           - Generate augmented variants for training data
           - Save in YOLO format
    
    Processing Strategy:
        - Training: Save clean (resized) + augmented versions
        - Validation/Test: Save clean (resized) only
    
    Output:
        Unified dataset at {target_path} with standardized structure
        ready for model training.
    """
    # Load master configuration
    with open("configs/base_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Apply global experiment seed so that any stochastic components
    # in data processing/augmentation are repeatable across runs.
    seed = get_seed_from_config(cfg)
    set_global_seed(seed)
    print(f"[Data Engine] Using global seed: {seed}")

    # Create output directory structure
    target_root = cfg['data_engine']['target_path']
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(target_root, 'labels', split), exist_ok=True)

    # Initialize augmentation pipeline
    augmentor = DataAugmentor(cfg['data_engine'])
    
    # Process each dataset defined in configuration
    for name, data_cfg in cfg['datasets'].items():
        print(f"\nProcessing Dataset: {name.upper()}")
        root = data_cfg['root']

        # Map configuration keys to standard split names
        splits = {
            'train': data_cfg.get('train_subdir', 'images/train'),
            'val': data_cfg.get('val_subdir', 'images/val'),
        }
        if 'test_subdir' in data_cfg:
            splits['test'] = data_cfg['test_subdir']

        # Process each split (train/val/test)
        for split_name, rel_path in splits.items():
            img_dir = os.path.join(root, rel_path)
            if not os.path.exists(img_dir):
                continue

            # Infer corresponding label directory
            # Standard structure: root/images/split -> root/labels/split
            lbl_rel_path = rel_path.replace('images', 'labels')
            lbl_dir = os.path.join(root, lbl_rel_path)

            # Get all image files in current split
            images = [x for x in os.listdir(img_dir) 
                     if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"  -> Merging {split_name} ({len(images)} images)...")

            for img_file in tqdm(images):
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")

                # Process and save clean (resized-only) version
                # This is done for all splits to ensure standardized dimensions
                orig_img, orig_bboxes, orig_cls = augmentor.process_resize_only(
                    img_path, lbl_path
                )
                
                if orig_img is None:
                    continue

                # Prefix with dataset name to prevent filename collisions across datasets
                clean_name = f"{name}_{img_file}"
                save_yolo_data(
                    target_root, split_name, clean_name, 
                    orig_img, orig_bboxes, orig_cls
                )

                # Generate augmented variant for training data only
                if split_name == 'train' and cfg['data_engine']['augmentation']['enable']:
                    # Apply full augmentation pipeline (resize + stochastic transforms)
                    aug_img, aug_bboxes, aug_cls = augmentor.process(img_path, lbl_path)
                    
                    if aug_img is not None:
                        aug_name = f"{name}_aug_{img_file}"
                        save_yolo_data(
                            target_root, split_name, aug_name, 
                            aug_img, aug_bboxes, aug_cls
                        )

    print(f"\nSUCCESS. Unified dataset saved to: {target_root}")

if __name__ == "__main__":
    main()