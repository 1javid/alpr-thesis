"""
Evaluation Module - ALPR Object Detection System

Reads test images from a folder and ground truth bounding boxes from a CSV file,
runs inference with a trained model, and computes standard object detection metrics:
Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95.

Supports YOLOv11, YOLOv10, and RT-DETRv2 via Ultralytics.

Usage:
    python evaluate.py --model yolov11 \\
                       --weights runs/yolov11_run/weights/best.pt \\
                       --images path/to/test/images \\
                       --labels path/to/labels.csv

    python evaluate.py --model yolov10 \\
                       --weights runs/yolov10_run/weights/best.pt \\
                       --images path/to/test/images \\
                       --labels path/to/labels.csv

    python evaluate.py --model rtdetrv2 \\
                       --weights runs/rtdetrv2_run/weights/best.pt \\
                       --images path/to/test/images \\
                       --labels path/to/labels.csv

CSV Requirements:
    Must contain columns: file_name, xmin, ymin, xmax, ymax (pixel coordinates).
    Additional columns are ignored.

Output:
    runs/evaluation/{model}/metrics_summary.json  -- aggregate metrics
    runs/evaluation/{model}/per_image_results.csv -- per-image TP/FP/FN/precision/recall

Author: ALPR Thesis Project
"""

import argparse
import os
import json
import csv
import sys
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict

sys.path.append(os.getcwd())


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path):
    """
    Load ground truth bounding boxes from a CSV file.

    Args:
        csv_path (str): Path to CSV file with columns:
            file_name, xmin, ymin, xmax, ymax (pixel coordinates).
            Additional columns are silently ignored.

    Returns:
        dict: {file_name (str): [[xmin, ymin, xmax, ymax], ...]}
    """
    df = pd.read_csv(csv_path)
    required = {"file_name", "xmin", "ymin", "xmax", "ymax"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    gt = defaultdict(list)
    for _, row in df.iterrows():
        box = [float(row["xmin"]), float(row["ymin"]),
               float(row["xmax"]), float(row["ymax"])]
        gt[str(row["file_name"])].append(box)

    return dict(gt)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_type, weights_path):
    """
    Load a trained Ultralytics model.

    Args:
        model_type (str): One of 'yolov11', 'yolov10', or 'rtdetrv2'.
        weights_path (str): Path to trained weights (.pt file).

    Returns:
        Ultralytics model instance.
    """
    if model_type in ("yolov11", "yolov10"):
        from ultralytics import YOLO
        return YOLO(weights_path)
    elif model_type == "rtdetrv2":
        from ultralytics import RTDETR
        return RTDETR(weights_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, img_path):
    """
    Run inference on a single image.

    Args:
        model: Loaded Ultralytics model.
        img_path (str): Path to image file.

    Returns:
        list of [conf, xmin, ymin, xmax, ymax] (pixel coordinates, float).
        Returns empty list if image cannot be loaded or no detections.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Warning: could not load image: {img_path}")
        return []

    results = model(img, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # [N, 4] xmin ymin xmax ymax
        confs = result.boxes.conf.cpu().numpy()         # [N]

        for box, conf in zip(boxes_xyxy, confs):
            detections.append([float(conf),
                                float(box[0]), float(box[1]),
                                float(box[2]), float(box[3])])

    return detections


# ---------------------------------------------------------------------------
# IoU and matching
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """
    Compute Intersection over Union for two [xmin, ymin, xmax, ymax] boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def match_predictions(preds, gt_boxes, iou_threshold):
    """
    Greedily match predictions to ground truth boxes at a given IoU threshold.
    Predictions are processed in descending confidence order.
    Each GT box can be matched at most once.

    Args:
        preds (list): [[conf, xmin, ymin, xmax, ymax], ...] sorted desc by conf.
        gt_boxes (list): [[xmin, ymin, xmax, ymax], ...]
        iou_threshold (float): IoU threshold for a valid match.

    Returns:
        list of dicts: [{'conf': float, 'tp': bool}, ...]
            tp=True  → prediction matched a GT box (True Positive)
            tp=False → prediction did not match any GT (False Positive)
    """
    matched_gt = set()
    results = []

    # Sort by confidence descending
    sorted_preds = sorted(preds, key=lambda x: x[0], reverse=True)

    for pred in sorted_preds:
        conf = pred[0]
        pred_box = pred[1:]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            results.append({"conf": conf, "tp": True})
        else:
            results.append({"conf": conf, "tp": False})

    return results


# ---------------------------------------------------------------------------
# AP / mAP computation
# ---------------------------------------------------------------------------

def compute_ap_from_matches(all_matches, total_gt):
    """
    Compute Average Precision (AP) from a list of prediction matches.

    Args:
        all_matches (list): [{'conf': float, 'tp': bool}, ...] across all images.
        total_gt (int): Total number of ground truth boxes across all images.

    Returns:
        float: AP value in [0, 1].
    """
    if total_gt == 0:
        return 0.0

    sorted_matches = sorted(all_matches, key=lambda x: x["conf"], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for m in sorted_matches:
        if m["tp"]:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        prec = tp_cumsum / (tp_cumsum + fp_cumsum)
        rec = tp_cumsum / total_gt
        precisions.append(prec)
        recalls.append(rec)

    # Area under the precision-recall curve (monotone envelope)
    recalls_arr = np.concatenate([[0.0], recalls, [1.0]])
    precisions_arr = np.concatenate([[1.0], precisions, [0.0]])

    # Monotonically decrease precision
    for i in range(len(precisions_arr) - 2, -1, -1):
        precisions_arr[i] = max(precisions_arr[i], precisions_arr[i + 1])

    change_idx = np.where(recalls_arr[1:] != recalls_arr[:-1])[0] + 1
    ap = float(np.sum(
        (recalls_arr[change_idx] - recalls_arr[change_idx - 1]) * precisions_arr[change_idx]
    ))
    return ap


def compute_map(all_preds_per_image, all_gt_per_image, iou_thresholds):
    """
    Compute mean Average Precision over multiple IoU thresholds.

    Args:
        all_preds_per_image (dict): {filename: [[conf, xmin, ymin, xmax, ymax], ...]}
        all_gt_per_image (dict): {filename: [[xmin, ymin, xmax, ymax], ...]}
        iou_thresholds (list): IoU thresholds to average over.

    Returns:
        dict: {iou_threshold: ap_value, ...}
    """
    ap_per_threshold = {}

    for iou_thr in iou_thresholds:
        all_matches = []
        total_gt = 0

        for filename, preds in all_preds_per_image.items():
            gt_boxes = all_gt_per_image.get(filename, [])
            total_gt += len(gt_boxes)
            matches = match_predictions(preds, gt_boxes, iou_thr)
            all_matches.extend(matches)

        # Images with GT but no predictions contribute only FN (no matches added)
        for filename, gt_boxes in all_gt_per_image.items():
            if filename not in all_preds_per_image:
                total_gt += len(gt_boxes)

        ap = compute_ap_from_matches(all_matches, total_gt)
        ap_per_threshold[round(iou_thr, 2)] = ap

    return ap_per_threshold


# ---------------------------------------------------------------------------
# Per-image summary
# ---------------------------------------------------------------------------

def per_image_stats(preds, gt_boxes, iou_threshold=0.5):
    """
    Compute TP, FP, FN for a single image at IoU threshold 0.5.

    Returns:
        dict with keys: tp, fp, fn, precision, recall
    """
    matches = match_predictions(preds, gt_boxes, iou_threshold)
    tp = sum(1 for m in matches if m["tp"])
    fp = len(matches) - tp
    fn = max(0, len(gt_boxes) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(model_type, weights_path, images_dir, labels_csv, output_dir, conf_threshold=0.25):
    """
    Full evaluation pipeline.

    Args:
        model_type (str): 'yolov11', 'yolov10', or 'rtdetrv2'.
        weights_path (str): Path to trained model weights.
        images_dir (str): Directory containing test images.
        labels_csv (str): CSV with file_name, xmin, ymin, xmax, ymax columns.
        output_dir (str): Directory to save evaluation results.
        conf_threshold (float): Minimum confidence to keep a prediction.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading ground truth from: {labels_csv}")
    ground_truth = load_ground_truth(labels_csv)

    print(f"Loading model: {model_type} from {weights_path}")
    model = load_model(model_type, weights_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ])

    if not image_files:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_files)} images. Running inference...\n")

    all_preds_per_image = {}
    all_gt_per_image = {}
    per_image_rows = []
    skipped = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        if img_file not in ground_truth:
            print(f"  Skipping {img_file}: no ground truth entry in CSV.")
            skipped += 1
            continue

        gt_boxes = ground_truth[img_file]
        detections = run_inference(model, img_path)

        # Filter by confidence threshold
        preds = [d for d in detections if d[0] >= conf_threshold]

        all_preds_per_image[img_file] = preds
        all_gt_per_image[img_file] = gt_boxes

        stats = per_image_stats(preds, gt_boxes, iou_threshold=0.5)
        per_image_rows.append({
            "file_name": img_file,
            "num_gt": len(gt_boxes),
            "num_pred": len(preds),
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
        })

    print(f"\nMatched {len(per_image_rows)} images ({skipped} skipped, no GT entry).")

    # Compute aggregate metrics at IoU 0.5
    iou_05_threshold = 0.5
    total_tp = sum(r["tp"] for r in per_image_rows)
    total_fp = sum(r["fp"] for r in per_image_rows)
    total_fn = sum(r["fn"] for r in per_image_rows)
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)
              if (agg_precision + agg_recall) > 0 else 0.0)

    # mAP@0.5 and mAP@0.5:0.95
    iou_thresholds_map50 = [0.5]
    iou_thresholds_map5095 = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    ap_map50 = compute_map(all_preds_per_image, all_gt_per_image, iou_thresholds_map50)
    ap_map5095 = compute_map(all_preds_per_image, all_gt_per_image, iou_thresholds_map5095)

    map50 = ap_map50[0.5]
    map5095 = float(np.mean(list(ap_map5095.values())))

    summary = {
        "model": model_type,
        "weights": weights_path,
        "images_dir": images_dir,
        "labels_csv": labels_csv,
        "conf_threshold": conf_threshold,
        "num_images_evaluated": len(per_image_rows),
        "num_images_skipped": skipped,
        "total_gt_boxes": sum(r["num_gt"] for r in per_image_rows),
        "total_pred_boxes": sum(r["num_pred"] for r in per_image_rows),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(agg_precision, 4),
        "recall": round(agg_recall, 4),
        "f1": round(agg_f1, 4),
        "mAP@0.5": round(map50, 4),
        "mAP@0.5:0.95": round(map5095, 4),
        "AP_per_IoU_threshold": {str(k): round(v, 4) for k, v in ap_map5095.items()},
    }

    # Save metrics summary
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-image results
    per_image_path = os.path.join(output_dir, "per_image_results.csv")
    fieldnames = ["file_name", "num_gt", "num_pred", "tp", "fp", "fn", "precision", "recall"]
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print("\n" + "=" * 50)
    print(f"  Model        : {model_type}")
    print(f"  Images       : {len(per_image_rows)}")
    print(f"  Total GT     : {summary['total_gt_boxes']}")
    print(f"  Total Pred   : {summary['total_pred_boxes']}")
    print(f"  TP / FP / FN : {total_tp} / {total_fp} / {total_fn}")
    print(f"  Precision    : {summary['precision']}")
    print(f"  Recall       : {summary['recall']}")
    print(f"  F1           : {summary['f1']}")
    print(f"  mAP@0.5      : {summary['mAP@0.5']}")
    print(f"  mAP@0.5:0.95 : {summary['mAP@0.5:0.95']}")
    print("=" * 50)
    print(f"\nResults saved to: {output_dir}")
    print(f"  {summary_path}")
    print(f"  {per_image_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ALPR detection model against CSV ground truth."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov11", "yolov10", "rtdetrv2"],
        help="Model architecture: 'yolov11', 'yolov10', or 'rtdetrv2'",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., runs/yolov11_run/weights/best.pt)",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing test images",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to CSV file with ground truth bounding boxes "
             "(must contain: file_name, xmin, ymin, xmax, ymax)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save evaluation results "
             "(default: runs/evaluation/{model})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)",
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join("runs", "evaluation", args.model)

    evaluate(
        model_type=args.model,
        weights_path=args.weights,
        images_dir=args.images,
        labels_csv=args.labels,
        output_dir=output_dir,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
