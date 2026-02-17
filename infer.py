"""
Inference Module - ALPR Object Detection System

Provides unified inference interface for multiple object detection models.
Supports YOLOv10, YOLOv11, and RT-DETRv2 with consistent output format,
automatic visualization, and organized result storage.
All models use the Ultralytics framework for consistency.

Features:
    - Multi-model support (YOLOv10, YOLOv11, RT-DETRv2)
    - Single image or batch directory inference
    - Automatic bounding box visualization with confidence scores
    - Model-specific output directories (organized results)
    - GPU acceleration support

Usage:
    # YOLOv11 inference
    python infer.py --model yolov11 --weights runs/yolov11_run/weights/best.pt --source test.jpg
    
    # YOLOv10 inference
    python infer.py --model yolov10 --weights runs/yolov10_run/weights/best.pt --source test.jpg
    
    # RT-DETRv2 inference
    python infer.py --model rtdetrv2 --weights runs/rtdetrv2_run/weights/best.pt --source test.jpg

Output Directories:
    - runs/inference/yolov11/    (YOLOv11 results)
    - runs/inference/yolov10/    (YOLOv10 results)
    - runs/inference/rtdetrv2/   (RT-DETR results)

Author: ALPR Thesis Project
"""

import argparse
import os
import cv2
import yaml
import sys

# Setup module paths for imports
sys.path.append(os.getcwd())

class InferenceEngine:
    """
    Unified inference engine for multiple object detection models.
    
    This class provides a consistent interface for running inference with
    different model architectures using the Ultralytics framework.
    
    Attributes:
        args (Namespace): Command-line arguments containing model config
        output_dir (str): Directory for saving inference results
        model: Loaded Ultralytics model instance
    
    Supported Models:
        - yolov11: YOLOv11 (Ultralytics)
        - yolov10: YOLOv10 (Ultralytics)
        - rtdetrv2: RT-DETRv2 (Ultralytics)
    """
    
    def __init__(self, args):
        """
        Initialize inference engine with configuration.
        
        Args:
            args (Namespace): Parsed command-line arguments containing:
                - model: Model type ('yolov11', 'yolov10', or 'rtdetrv2')
                - weights: Path to trained model weights
                - source: Path to image or directory for inference
        """
        self.args = args
        # Create model-specific output directory
        self.output_dir = os.path.join("runs", "inference", args.model)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Inference results will be saved to: {self.output_dir}")

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """
        Load trained model using Ultralytics framework.
        
        All models (YOLOv11, YOLOv10, RT-DETRv2) use the same loading procedure
        with Ultralytics for consistency.
        
        Returns:
            Loaded Ultralytics model instance ready for inference
        
        Note:
            Models automatically handle device placement and evaluation mode.
        """
        print(f"Loading {self.args.model} from {self.args.weights}...")
        
        if self.args.model in ['yolov10', 'yolov11']:
            from ultralytics import YOLO
            return YOLO(self.args.weights)

        elif self.args.model == 'rtdetrv2':
            from ultralytics import RTDETR
            return RTDETR(self.args.weights)

    def predict(self, img_path):
        """
        Run inference on a single image.
        
        Uses Ultralytics' built-in preprocessing, inference, and visualization
        for all model types (YOLOv11, YOLOv10, RT-DETRv2).
        
        Args:
            img_path (str): Path to input image file
        
        Processing Pipeline:
            1. Run inference (Ultralytics handles preprocessing internally)
            2. Generate visualization with predictions
            3. Save annotated image
        
        Note:
            If image cannot be loaded, the function returns silently without error.
        """
        # Load input image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            return

        # Prepare output path
        save_path = os.path.join(self.output_dir, os.path.basename(img_path))
        
        # Run inference (Ultralytics handles preprocessing internally)
        results = self.model(img)
        
        # Generate visualization with predictions
        res_plotted = results[0].plot()
        
        # Save annotated image
        cv2.imwrite(save_path, res_plotted)

        print(f"Saved prediction to: {save_path}")

def main():
    """
    Main inference entry point.
    
    Parses command-line arguments, initializes inference engine,
    and processes single image or batch of images.
    
    Command-line Arguments:
        --model: Model type ('yolov10', 'yolov11', or 'rtdetrv2')
        --weights: Path to trained model weights file
        --source: Path to image file or directory of images
    
    Behavior:
        - Single image: Runs inference on one image
        - Directory: Processes all .jpg and .png files in directory
        - Results saved to: runs/inference/{model}/
    
    Examples:
        # YOLOv11 inference
        python infer.py --model yolov11 --weights runs/yolov11_run/weights/best.pt --source test.jpg
        
        # YOLOv10 inference
        python infer.py --model yolov10 --weights runs/yolov10_run/weights/best.pt --source test.jpg
        
        # RT-DETRv2 inference
        python infer.py --model rtdetrv2 --weights runs/rtdetrv2_run/weights/best.pt --source test.jpg
        
        # Batch inference on directory
        python infer.py --model yolov11 --weights runs/yolov11_run/weights/best.pt --source test_images/
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Unified Object Detection Inference")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['yolov10', 'yolov11', 'rtdetrv2'],
        help="Model architecture: 'yolov10', 'yolov11', or 'rtdetrv2'"
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        required=True, 
        help="Path to trained model weights (e.g., best.pt or checkpoint.pth)"
    )
    parser.add_argument(
        '--source', 
        type=str, 
        required=True, 
        help="Path to input image file or directory containing images"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(args)

    # Process source (single image or directory)
    if os.path.isdir(args.source):
        # Batch processing: process all images in directory
        print(f"Processing directory: {args.source}")
        files = [
            os.path.join(args.source, f) 
            for f in os.listdir(args.source) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]
        print(f"Found {len(files)} images")
        
        for f in files:
            engine.predict(f)
    else:
        # Single image processing
        print(f"Processing image: {args.source}")
        engine.predict(args.source)
    
    print(f"\nInference complete. Results saved to: {engine.output_dir}")

if __name__ == "__main__":
    main()