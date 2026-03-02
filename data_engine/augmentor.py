"""
Data Augmentor Module

Provides image augmentation capabilities for object detection datasets using Albumentations.
Supports both training augmentation (with stochastic transformations) and resize-only
processing for validation/test sets.

Key Features:
    - Standardized image resizing
    - Brightness/contrast adjustments
    - Blur effects
    - Mild geometric transformations (perspective, crop, shear, shift/scale/rotate)
    - Bounding box preservation during augmentation

Author: ALPR Thesis Project
"""

import cv2
import albumentations as A
import numpy as np
import os

class DataAugmentor:
    """
    Image augmentation handler for object detection datasets.
    
    This class provides two processing modes:
    1. Full augmentation: Resize + stochastic augmentations (for training)
    2. Resize-only: Standardized resizing without augmentation (for validation/test)
    
    Attributes:
        target_size (int): Target image dimension (square resize)
        transform (A.Compose): Albumentations transformation pipeline
    """
    
    def __init__(self, cfg):
        """
        Initialize augmentation pipeline based on configuration.
        
        Args:
            cfg (dict): Data engine configuration containing:
                - img_size: Target image size for resize
                - augmentation.enable: Whether to enable augmentations
                - augmentation.prob: Global augmentation probability
                - augmentation.params: Specific augmentation parameters
        
        Note:
            All augmentations are compatible with bounding boxes in YOLO format
            (normalized [x_center, y_center, width, height]).
        """
        self.target_size = cfg.get('img_size', 640)
        params = cfg['augmentation']['params']
        p = cfg['augmentation']['prob']

        # Build augmentation pipeline
        transforms = []
        
        # Always resize to target dimensions (ensures consistent input size)
        transforms.append(A.Resize(height=self.target_size, width=self.target_size))

        # Add stochastic augmentations if enabled
        if cfg['augmentation']['enable']:
            # Brightness and contrast variations (simulates different lighting conditions).
            # Use small limits (±0.1) and moderate probability (0.2).
            if params.get('brightness_contrast'):
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.2,
                    )
                )
            
            # Blur effect (simulates mild motion blur or camera focus issues)
            if params.get('blur_limit'):
                transforms.append(
                    A.Blur(
                        blur_limit=params.get('blur_limit'),
                        p=0.1,
                    )
                )

            # Perspective distortion (simulates slight viewpoint changes)
            if params.get('perspective'):
                transforms.append(
                    A.Perspective(
                        scale=(0.02, 0.08),  # slight perspective changes
                        p=0.2,
                    )
                )

            # Random crop with resize back to target size (light crop)
            if params.get('random_crop'):
                transforms.append(
                    A.RandomResizedCrop(
                        height=self.target_size,
                        width=self.target_size,
                        scale=(0.85, 1.0),   # keep at least 85% of area
                        ratio=(0.9, 1.1),
                        p=0.2,
                    )
                )

            # Shear via affine transformation (simulates mild camera skew)
            if params.get('shear'):
                transforms.append(
                    A.Affine(
                        shear=(-5, 5),      # small shear in degrees
                        p=0.2,
                    )
                )

            # Geometric transformations (simulates small shifts, scales, rotations)
            # shift_limit: Translation up to ~6% of image dimensions
            # scale_limit: Scaling by ±10%
            # rotate_limit: Rotation up to ±10 degrees
            if params.get('shift_scale_rotate', True):
                transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, 
                        scale_limit=0.1, 
                        rotate_limit=10, 
                        p=0.2,
                    )
                )

        # Compose all transformations with bounding box support
        # Uses YOLO format: normalized [x_center, y_center, width, height]
        # min_visibility=0.3: Drops bboxes if less than 30% visible after transformation
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo', 
                min_visibility=0.3, 
                label_fields=['class_labels']
            )
        )

    def process(self, img_path, label_path):
        """
        Process image with full augmentation pipeline.
        
        Applies resize and stochastic augmentations to an image while preserving
        and transforming its bounding boxes.
        
        Args:
            img_path (str): Path to input image file
            label_path (str): Path to YOLO-format label file (one line per object)
        
        Returns:
            tuple: (augmented_image, augmented_bboxes, class_labels)
                - augmented_image: RGB numpy array of augmented image
                - augmented_bboxes: List of bboxes in YOLO format [xc, yc, w, h]
                - class_labels: List of class IDs corresponding to bboxes
                Returns (None, None, None) if image cannot be loaded or augmentation fails
        
        Note:
            This method should be used for training data augmentation.
            For validation/test, use process_resize_only() instead.
        """
        # Read and convert image to RGB
        image = cv2.imread(img_path)
        if image is None:
            return None, None, None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse YOLO-format labels
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # YOLO format: x_center, y_center, width, height (all normalized)
                    bbox = [float(x) for x in parts[1:5]]
                    bboxes.append(bbox)
                    class_labels.append(cls_id)

        # Apply augmentation pipeline
        try:
            transformed = self.transform(
                image=image, 
                bboxes=bboxes, 
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            print(f"Augmentation failed for {img_path}: {e}")
            return None, None, None

    def process_resize_only(self, img_path, label_path):
        """
        Process image with resize-only (no stochastic augmentation).
        
        Applies only standardized resizing without any random augmentations.
        This ensures consistent preprocessing while maintaining original image
        characteristics.
        
        Intended use cases:
            - All validation/test images (no augmentation needed)
            - Clean training copies before generating augmented variants
        
        Args:
            img_path (str): Path to input image file
            label_path (str): Path to YOLO-format label file
        
        Returns:
            tuple: (resized_image, resized_bboxes, class_labels)
                - resized_image: RGB numpy array of resized image
                - resized_bboxes: List of bboxes adjusted for new size (YOLO format)
                - class_labels: List of class IDs corresponding to bboxes
                Returns (None, None, None) if image cannot be loaded or processing fails
        
        Note:
            Uses min_visibility=0.0 to preserve all bboxes during resize
            (no filtering based on visibility threshold).
        """
        # Read and convert image to RGB
        image = cv2.imread(img_path)
        if image is None:
            return None, None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse YOLO-format labels
        bboxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bboxes.append(bbox)
                    class_labels.append(cls_id)

        # Create resize-only transformation pipeline
        # min_visibility=0.0 ensures all bboxes are kept during resize
        resize_tfm = A.Compose(
            [A.Resize(height=self.target_size, width=self.target_size)],
            bbox_params=A.BboxParams(
                format='yolo', 
                min_visibility=0.0, 
                label_fields=['class_labels']
            )
        )

        # Apply resize transformation
        try:
            transformed = resize_tfm(
                image=image, 
                bboxes=bboxes, 
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            print(f"Resize-only processing failed for {img_path}: {e}")
            return None, None, None