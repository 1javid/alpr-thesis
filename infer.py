"""
Inference Module - ALPR Object Detection System

Provides unified inference interface for multiple object detection models.
Supports both YOLO-based models (YOLOv10, YOLOv11) and RT-DETRv2 with
consistent output format and visualization.

Features:
    - Multi-model support (YOLO, RT-DETRv2)
    - Single image or batch directory inference
    - Automatic bounding box visualization
    - Confidence score filtering
    - GPU acceleration support

Usage:
    # YOLO inference
    python infer.py --model yolo --weights runs/yolov11_run/weights/best.pt --source test_image.jpg
    
    # RT-DETRv2 inference
    python infer.py --model rtdetrv2 --weights checkpoint.pth --model_config config.yaml --source test_dir/

Author: ALPR Thesis Project
"""

import argparse
import os
import cv2
import torch
import numpy as np
import yaml
import sys

# Setup module paths for imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'models/rtdetr_pytorch'))

def draw_predictions(img, boxes, scores, labels, class_names, threshold=0.5):
    """
    Draw bounding boxes and labels on image.
    
    Visualizes object detection predictions by drawing colored bounding boxes
    with class names and confidence scores.
    
    Args:
        img (np.ndarray): Input image in BGR format
        boxes (np.ndarray): Bounding boxes in [x1, y1, x2, y2] format (pixels)
        scores (np.ndarray): Confidence scores for each detection
        labels (np.ndarray): Class label IDs for each detection
        class_names (dict): Mapping from class ID to class name
        threshold (float): Confidence threshold for displaying predictions
    
    Returns:
        np.ndarray: Image with drawn predictions
    
    Note:
        Predictions below the threshold are filtered out.
        Boxes are drawn in green with class name and confidence score.
    """
    img_draw = img.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        # Filter low-confidence predictions
        if score < threshold:
            continue
        
        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Format label text with class name and confidence
        class_name = class_names.get(int(label), 'Unknown')
        label_text = f"{class_name} {score:.2f}"
        
        # Draw bounding box (green, 2px thickness)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label text above box
        cv2.putText(
            img_draw, 
            label_text, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )
    
    return img_draw

class InferenceEngine:
    """
    Unified inference engine for multiple object detection models.
    
    This class provides a consistent interface for running inference with
    different model architectures (YOLO, RT-DETRv2) while handling
    model-specific preprocessing and postprocessing internally.
    
    Attributes:
        args (Namespace): Command-line arguments containing model config
        output_dir (str): Directory for saving inference results
        classes (dict): Class ID to class name mapping
        model: Loaded model instance (type depends on model architecture)
    
    Supported Models:
        - yolo: Ultralytics YOLO models (YOLOv10, YOLOv11)
        - rtdetrv2: RT-DETRv2 (Official PyTorch implementation)
    """
    
    def __init__(self, args):
        """
        Initialize inference engine with configuration.
        
        Args:
            args (Namespace): Parsed command-line arguments containing:
                - model: Model type ('yolo' or 'rtdetrv2')
                - weights: Path to trained model weights
                - source: Path to image or directory for inference
                - model_config: (RT-DETRv2 only) Path to model config YAML
        """
        self.args = args
        self.output_dir = "runs/inference"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load class names from master configuration
        with open("configs/base_config.yaml") as f:
            cfg = yaml.safe_load(f)
        self.classes = cfg['classes']

        # Load model based on type
        self.model = self._load_model()

    def _load_model(self):
        """
        Load trained model based on architecture type.
        
        Handles model-specific loading procedures including:
        - Ultralytics YOLO: Direct weight loading
        - RT-DETRv2: Config-based architecture reconstruction + checkpoint loading
        
        Returns:
            Loaded model instance ready for inference
        
        Raises:
            ValueError: If RT-DETRv2 is selected without model_config
            FileNotFoundError: If weights or config file not found
        
        Note:
            RT-DETRv2 models are automatically moved to GPU and set to eval mode.
            YOLO models handle device placement internally.
        """
        print(f"Loading {self.args.model} from {self.args.weights}...")
        
        if self.args.model == 'yolo':
            # Load Ultralytics YOLO model (supports YOLOv10, YOLOv11)
            from ultralytics import YOLO
            return YOLO(self.args.weights)

        elif self.args.model == 'rtdetrv2':
            # Load RT-DETRv2 (Official PyTorch implementation)
            from src.core import YAMLConfig
            
            # RT-DETRv2 requires architecture config for model reconstruction
            # If not provided, load from base_config.yaml
            if not self.args.model_config:
                print("No --model_config provided, loading from base_config.yaml...")
                with open("configs/base_config.yaml") as f:
                    base_cfg = yaml.safe_load(f)
                model_config_path = base_cfg['models']['rtdetrv2']['config_file']
                print(f"Using RT-DETR config: {model_config_path}")
            else:
                model_config_path = self.args.model_config
                # Still need to load base_config for num_classes
                with open("configs/base_config.yaml") as f:
                    base_cfg = yaml.safe_load(f)
            
            # Load RT-DETR config WITHOUT building model yet
            cfg = YAMLConfig(model_config_path, resume=None)
            
            # CRITICAL: Override num_classes BEFORE building model
            # The config defaults to 80 (COCO), but trained model may have different number
            num_classes = len(base_cfg['classes'])
            cfg.yaml_cfg['num_classes'] = num_classes
            print(f"Setting num_classes to {num_classes} (from base_config.yaml)")
            
            # Now build model with correct number of classes
            model = cfg.model
            
            # Load trained weights from checkpoint
            checkpoint = torch.load(self.args.weights, map_location='cuda:0')
            
            # Extract state dict (handle EMA if present)
            if 'ema' in checkpoint:
                # Use Exponential Moving Average weights if available (often better)
                state_dict = checkpoint['ema']['module']
            else:
                # Fall back to standard model weights
                state_dict = checkpoint['model']
            
            # Load weights and prepare for inference
            model.load_state_dict(state_dict)
            model.cuda().eval()
            return model

    def predict(self, img_path):
        """
        Run inference on a single image.
        
        Performs model-specific preprocessing, inference, postprocessing,
        and visualization. Saves the annotated result to the output directory.
        
        Args:
            img_path (str): Path to input image file
        
        Processing Pipeline:
            1. Load image from disk
            2. Apply model-specific preprocessing
            3. Run inference
            4. Postprocess predictions
            5. Visualize results
            6. Save annotated image
        
        Model-Specific Handling:
            YOLO:
                - Uses built-in Ultralytics preprocessing and visualization
                - Automatically handles device placement
            
            RT-DETRv2:
                - Manual preprocessing (RGB conversion, tensor transform)
                - Converts normalized center-format boxes to pixel corner-format
                - Custom visualization with draw_predictions()
        
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
        
        # === YOLO INFERENCE ===
        if self.args.model == 'yolo':
            # Run inference (Ultralytics handles preprocessing internally)
            results = self.model(img)
            
            # Generate visualization with predictions
            res_plotted = results[0].plot()
            
            # Save annotated image
            cv2.imwrite(save_path, res_plotted)

        # === RT-DETRv2 INFERENCE ===
        elif self.args.model == 'rtdetrv2':
            import torchvision.transforms as T
            
            # Preprocessing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            # Convert to tensor and add batch dimension
            img_tensor = T.ToTensor()(img_rgb).cuda()
            img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
            
            # Forward pass (no gradient computation needed)
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Postprocessing
            # RT-DETRv2 outputs:
            #   - pred_logits: [batch, num_queries, num_classes] (raw logits)
            #   - pred_boxes: [batch, num_queries, 4] (normalized cx, cy, w, h)
            pred_logits = output['pred_logits'][0]  # Remove batch dim
            pred_boxes = output['pred_boxes'][0]
            
            # Convert logits to probabilities and extract class predictions
            probs = pred_logits.sigmoid()
            scores, labels = probs.max(-1)  # Get max probability per query
            
            # Filter predictions by confidence threshold
            keep = scores > 0.5
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_boxes[keep]
            
            # Convert bounding box format
            # From: Normalized (cx, cy, w, h) where values are in [0, 1]
            # To: Absolute (x1, y1, x2, y2) in pixel coordinates
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
            boxes_abs[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
            boxes_abs[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
            boxes_abs[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2
            
            # Visualize predictions
            res_img = draw_predictions(
                img, 
                boxes_abs.cpu().numpy(), 
                scores.cpu().numpy(), 
                labels.cpu().numpy(), 
                self.classes
            )
            
            # Save annotated image
            cv2.imwrite(save_path, res_img)

        print(f"Saved prediction to: {save_path}")

def main():
    """
    Main inference entry point.
    
    Parses command-line arguments, initializes inference engine,
    and processes single image or batch of images.
    
    Command-line Arguments:
        --model: Model type ('yolo' or 'rtdetrv2')
        --weights: Path to trained model weights file
        --source: Path to image file or directory of images
        --model_config: (Optional) Path to RT-DETR config YAML. If not provided,
                        automatically loaded from configs/base_config.yaml
    
    Behavior:
        - Single image: Runs inference on one image
        - Directory: Processes all .jpg and .png files in directory
    
    Examples:
        # Single image with YOLO
        python infer.py --model yolo --weights runs/yolov11_run/weights/best.pt --source test.jpg
        
        # RT-DETRv2 (auto-loads config from base_config.yaml)
        python infer.py --model rtdetrv2 --weights runs/rtdetrv2_run/checkpoint_best.pth --source test.jpg
        
        # RT-DETRv2 with custom config (optional)
        python infer.py --model rtdetrv2 --weights checkpoint.pth \
                        --model_config path/to/custom_config.yaml --source test_images/
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Unified Object Detection Inference")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['yolo', 'rtdetrv2'],
        help="Model architecture: 'yolo' (YOLOv10/v11) or 'rtdetrv2'"
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
    parser.add_argument(
        '--model_config', 
        type=str, 
        default=None, 
        help="Path to model config YAML (optional for RT-DETRv2, uses base_config.yaml if not provided)"
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