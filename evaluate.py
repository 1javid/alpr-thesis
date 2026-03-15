"""
Evaluation Module - ALPR Object Detection System

Reads test images from a folder and ground truth bounding boxes from a CSV file,
runs inference with a trained model, and computes standard object detection metrics.

Outputs (no annotated images):
    metrics_summary.json         -- aggregate metrics (precision, recall, F1, mAP@0.5, mAP@0.5:0.95)
                                    and inference speed (mean latency ms, FPS)
    per_image_results.csv        -- per-image TP / FP / FN / precision / recall
    predictions_detailed.csv     -- one row per detection (TP/FP/FN) with GT and predicted
                                    bounding box coordinates, confidence score, and IoU
    pr_curve.png                 -- precision-recall curve
    f1_curve.png                 -- F1 score vs confidence threshold curve
    confusion_matrix.png         -- TP / FP / FN confusion matrix heatmap

Supports YOLOv11 and YOLOv26 via Ultralytics, and RF-DETR via the `rfdetr` library.

Usage:
    python evaluate.py --model yolov11 \\
                       --weights runs/yolov11_run/weights/best.pt \\
                       --images path/to/test/images \\
                       --labels path/to/labels.csv

CSV Requirements:
    Must contain columns:
        - file_name
        - xmin, ymin, xmax, ymax (pixel coordinates)
        - plates_count  (number of license plates in the image)
    Only images with plates_count == 1 are evaluated; others are ignored.

Author: ALPR Thesis Project
"""

import argparse
import os
import json
import csv
import sys
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.append(os.getcwd())


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path):
    """
    Load ground truth bounding boxes from a CSV file, together with plate counts.

    One row per bounding box. Multiple rows may share the same file_name
    (one per license plate in the image).

    Args:
        csv_path (str): Path to CSV with columns:
                        file_name, xmin, ymin, xmax, ymax, plates_count.
                        Additional columns are ignored.

    Returns:
        gt_boxes (dict): {file_name: [[xmin, ymin, xmax, ymax], ...]}
        plate_counts (dict): {file_name: int}
    """
    df = pd.read_csv(csv_path)
    required = {"file_name", "xmin", "ymin", "xmax", "ymax", "plates_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    gt = defaultdict(list)
    plate_counts = {}
    skipped = 0
    for _, row in df.iterrows():
        try:
            fname = str(row["file_name"])
            box = [
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            ]
            count = int(row["plates_count"])
        except (ValueError, TypeError):
            skipped += 1
            continue
        gt[fname].append(box)
        plate_counts[fname] = count

    if skipped:
        print(f"  Warning: skipped {skipped} CSV rows with missing or invalid values.")

    return dict(gt), plate_counts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_type, weights_path):
    """
    Load a trained Ultralytics model.

    Args:
        model_type (str): 'yolov11', 'yolov26', or 'rfdetr'.
        weights_path (str): Path to trained weights (YOLO .pt or RF-DETR checkpoint).

    Returns:
        Ultralytics model instance.
    """
    if model_type in ("yolov11", "yolov26"):
        from ultralytics import YOLO
        return YOLO(weights_path)
    elif model_type == "rfdetr":
        from rfdetr import RFDETRSmall
        return RFDETRSmall(pretrain_weights=weights_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, img_path, model_type):
    """
    Run inference on a single image.

    Args:
        model: Loaded model (Ultralytics YOLO or RF-DETR).
        img_path (str): Path to image file.

    Returns:
        list of [conf, xmin, ymin, xmax, ymax] in pixel coordinates.
        Returns empty list if image cannot be loaded or no detections.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Warning: could not load image: {img_path}")
        return []

    detections = []

    if model_type in ("yolov11", "yolov26"):
        results = model(img, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes_xyxy, confs):
                detections.append(
                    [
                        float(conf),
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ]
                )
    elif model_type == "rfdetr":
        # RF-DETR uses the `rfdetr` API and returns a supervision.Detections object.
        # We adapt it to the same [conf, xmin, ymin, xmax, ymax] format.
        try:
            dets = model.predict(img_path)
        except TypeError:
            # Some versions expect explicit threshold arg; fall back gracefully.
            dets = model.predict(img_path, threshold=0.0)

        # Supervision Detections typically expose `.xyxy` and `.confidence`.
        xyxy = getattr(dets, "xyxy", None)
        confs = getattr(dets, "confidence", None)
        if xyxy is None or confs is None:
            print("  Warning: RF-DETR predict() did not return xyxy/confidence as expected.")
            return []

        boxes_xyxy = np.array(xyxy)
        confs = np.array(confs)
        for box, conf in zip(boxes_xyxy, confs):
            detections.append(
                [
                    float(conf),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ]
            )

    return detections


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Compute IoU between two [xmin, ymin, xmax, ymax] pixel-coordinate boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Detailed matching
# ---------------------------------------------------------------------------

def match_predictions_detailed(preds, gt_boxes, iou_threshold):
    """
    Greedily match predictions to ground truth boxes in descending confidence order.
    Each GT box is matchable at most once.

    Args:
        preds (list): [[conf, xmin, ymin, xmax, ymax], ...]
        gt_boxes (list): [[xmin, ymin, xmax, ymax], ...]
        iou_threshold (float): Minimum IoU for a valid match.

    Returns:
        tp_fp (list): One entry per prediction:
            {
                'conf':      float,
                'pred_box':  [xmin, ymin, xmax, ymax],
                'gt_box':    [xmin, ymin, xmax, ymax] or None,
                'iou':       float,
                'tp':        bool,
            }
        fn_boxes (list): GT boxes that were never matched:
            [[xmin, ymin, xmax, ymax], ...]
    """
    matched_gt = set()
    tp_fp = []

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
            tp_fp.append({
                "conf": conf,
                "pred_box": pred_box,
                "gt_box": gt_boxes[best_gt_idx],
                "iou": best_iou,
                "tp": True,
            })
        else:
            tp_fp.append({
                "conf": conf,
                "pred_box": pred_box,
                "gt_box": None,
                "iou": best_iou if best_gt_idx >= 0 else 0.0,
                "tp": False,
            })

    fn_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp_fp, fn_boxes


# ---------------------------------------------------------------------------
# AP / mAP (for curve + summary)
# ---------------------------------------------------------------------------

def build_pr_points(all_matches, total_gt):
    """
    Build precision-recall points by walking sorted matches.

    Args:
        all_matches (list): [{'conf': float, 'tp': bool}, ...] across all images.
        total_gt (int): Total GT boxes across all images.

    Returns:
        confs, precisions, recalls (numpy arrays, sorted descending by conf).
    """
    if total_gt == 0 or not all_matches:
        return np.array([]), np.array([]), np.array([])

    sorted_m = sorted(all_matches, key=lambda x: x["conf"], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions, recalls, confs = [], [], []

    for m in sorted_m:
        if m["tp"]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)
        confs.append(m["conf"])

    return (np.array(confs),
            np.array(precisions),
            np.array(recalls))


def ap_from_pr(precisions, recalls):
    """Area under the monotone-envelope PR curve."""
    if len(precisions) == 0:
        return 0.0
    r = np.concatenate([[0.0], recalls, [1.0]])
    p = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0] + 1
    return float(np.sum((r[idx] - r[idx - 1]) * p[idx]))


def compute_map_across_thresholds(all_preds_per_image, all_gt_per_image, iou_thresholds):
    """
    Compute AP at each IoU threshold and return a dict {threshold: ap}.
    """
    ap_per_threshold = {}
    for iou_thr in iou_thresholds:
        all_matches = []
        total_gt = 0
        for filename, preds in all_preds_per_image.items():
            gt_boxes = all_gt_per_image.get(filename, [])
            total_gt += len(gt_boxes)
            tp_fp, _ = match_predictions_detailed(preds, gt_boxes, iou_thr)
            all_matches.extend({"conf": m["conf"], "tp": m["tp"]} for m in tp_fp)
        for filename, gt_boxes in all_gt_per_image.items():
            if filename not in all_preds_per_image:
                total_gt += len(gt_boxes)
        _, precs, recs = build_pr_points(all_matches, total_gt)
        ap_per_threshold[round(iou_thr, 2)] = ap_from_pr(precs, recs)
    return ap_per_threshold


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linestyle": "-",
}


def plot_pr_curve(precisions, recalls, ap50, model_type, output_path):
    """Save precision-recall curve as PNG."""
    # Monotone envelope
    r = np.concatenate([[0.0], recalls, [1.0]])
    p = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(r, p, color="#1f77b4", linewidth=2,
                label=f"license  (AP@0.5 = {ap50:.3f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — {model_type}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def plot_f1_curve(confs, precisions, recalls, model_type, output_path):
    """
    Save F1 vs confidence threshold curve as PNG.
    Sweeps confidence values, recomputes precision/recall by filtering predictions.
    """
    thresholds = np.linspace(0.0, 1.0, 200)
    f1_scores = []

    for thr in thresholds:
        mask = confs >= thr
        if mask.sum() == 0:
            f1_scores.append(0.0)
            continue
        p = precisions[mask][-1] if mask.any() else 0.0
        r = recalls[mask][-1] if mask.any() else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_conf = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds, f1_scores, color="#2ca02c", linewidth=2,
                label=f"license  (best F1={best_f1:.3f} @ conf={best_conf:.2f})")
        ax.axvline(best_conf, color="#d62728", linestyle="--", linewidth=1,
                   alpha=0.7, label=f"best conf = {best_conf:.2f}")
        ax.set_xlabel("Confidence Threshold", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"F1-Confidence Curve — {model_type}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def plot_confusion_matrix(tp, fp, fn, model_type, output_path, normalize=True):
    """
    Save a detection confusion matrix as PNG.

    Rows = actual class, columns = predicted class.
    For single-class detection the meaningful cells are TP, FP, FN.
    TN (background correctly ignored) is undefined and omitted.

        Predicted:   license   background
    Actual license:    TP          FN
    Actual background: FP          –
    """
    matrix = np.array([[tp, fn],
                        [fp, 0]], dtype=float)

    labels = ["license\n(actual)", "background\n(actual)"]
    col_labels = ["license\n(predicted)", "background\n(predicted)"]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = matrix / row_sums
        fmt = ".2f"
        title_suffix = "(normalised)"
    else:
        display = matrix
        fmt = "d"
        title_suffix = "(counts)"

    with plt.rc_context({**_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(display, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=1 if normalize else None)
        plt.colorbar(im, ax=ax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(f"Confusion Matrix — {model_type}\n{title_suffix}", fontsize=12)

        # Annotate cells
        thresh = display.max() / 2.0
        for i in range(2):
            for j in range(2):
                cell_label = f"{display[i, j]:{fmt}}"
                if i == 1 and j == 1:
                    cell_label = "–"
                ax.text(j, i, cell_label, ha="center", va="center",
                        fontsize=13,
                        color="white" if display[i, j] > thresh else "black")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Detailed predictions CSV
# ---------------------------------------------------------------------------

def save_predictions_csv(all_detailed, output_path):
    """
    Save a CSV with one row per detection event (TP, FP, or FN).

    Columns mirror the labels CSV format plus predicted box columns:
        file_name, match_type,
        gt_xmin, gt_ymin, gt_xmax, gt_ymax,
        pred_xmin, pred_ymin, pred_xmax, pred_ymax,
        confidence, iou

    match_type values:
        TP  — prediction matched a GT box
        FP  — prediction had no matching GT box
        FN  — GT box had no matching prediction
    """
    fieldnames = [
        "file_name", "match_type",
        "gt_xmin", "gt_ymin", "gt_xmax", "gt_ymax",
        "pred_xmin", "pred_ymin", "pred_xmax", "pred_ymax",
        "confidence", "iou",
    ]

    def _fmt(v):
        return round(v, 2) if v is not None else ""

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filename, tp_fp_list, fn_boxes in all_detailed:
            for m in tp_fp_list:
                match_type = "TP" if m["tp"] else "FP"
                gt = m["gt_box"]
                pb = m["pred_box"]
                writer.writerow({
                    "file_name": filename,
                    "match_type": match_type,
                    "gt_xmin":  _fmt(gt[0]) if gt else "",
                    "gt_ymin":  _fmt(gt[1]) if gt else "",
                    "gt_xmax":  _fmt(gt[2]) if gt else "",
                    "gt_ymax":  _fmt(gt[3]) if gt else "",
                    "pred_xmin": _fmt(pb[0]),
                    "pred_ymin": _fmt(pb[1]),
                    "pred_xmax": _fmt(pb[2]),
                    "pred_ymax": _fmt(pb[3]),
                    "confidence": _fmt(m["conf"]),
                    "iou": _fmt(m["iou"]),
                })

            for fn_box in fn_boxes:
                writer.writerow({
                    "file_name": filename,
                    "match_type": "FN",
                    "gt_xmin":  _fmt(fn_box[0]),
                    "gt_ymin":  _fmt(fn_box[1]),
                    "gt_xmax":  _fmt(fn_box[2]),
                    "gt_ymax":  _fmt(fn_box[3]),
                    "pred_xmin": "", "pred_ymin": "", "pred_xmax": "", "pred_ymax": "",
                    "confidence": "",
                    "iou": "",
                })


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(model_type, weights_path, images_dir, labels_csv,
             output_dir, conf_threshold=0.25):
    """
    Full evaluation pipeline: inference → matching → metrics → plots → CSVs.

    Args:
        model_type (str): 'yolov11', 'yolov26', or 'rfdetr'.
        weights_path (str): Path to trained model weights.
        images_dir (str): Directory containing test images.
        labels_csv (str): CSV with file_name, xmin, ymin, xmax, ymax columns.
        output_dir (str): Directory to save all evaluation outputs.
        conf_threshold (float): Minimum confidence to keep a prediction.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading ground truth from: {labels_csv}")
    ground_truth, plate_counts = load_ground_truth(labels_csv)

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
    all_detailed = []       # (filename, tp_fp_list, fn_boxes)
    per_image_rows = []
    all_matches_05 = []     # for PR/F1 curves at IoU=0.5
    total_gt_05 = 0
    skipped = 0
    inference_times_ms = []  # per-image inference latency for speed metrics

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        if img_file not in ground_truth:
            print(f"  Skipping {img_file}: no ground truth entry in CSV.")
            skipped += 1
            continue

        # Only consider images with exactly one license plate
        if plate_counts.get(img_file, 0) != 1:
            print(f"  Skipping {img_file}: plates_count={plate_counts.get(img_file, 0)} != 1.")
            skipped += 1
            continue

        gt_boxes = ground_truth[img_file]
        t0 = time.perf_counter()
        detections = run_inference(model, img_path, model_type)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        inference_times_ms.append(inference_ms)
        preds = [d for d in detections if d[0] >= conf_threshold]

        all_preds_per_image[img_file] = preds
        all_gt_per_image[img_file] = gt_boxes

        tp_fp_list, fn_boxes = match_predictions_detailed(preds, gt_boxes, iou_threshold=0.5)
        all_detailed.append((img_file, tp_fp_list, fn_boxes))

        tp = sum(1 for m in tp_fp_list if m["tp"])
        fp = len(tp_fp_list) - tp
        fn = len(fn_boxes)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_image_rows.append({
            "file_name": img_file,
            "num_gt": len(gt_boxes),
            "num_pred": len(preds),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "inference_ms": round(inference_ms, 2),
        })

        all_matches_05.extend({"conf": m["conf"], "tp": m["tp"]} for m in tp_fp_list)
        total_gt_05 += len(gt_boxes)

    print(f"Matched {len(per_image_rows)} images ({skipped} skipped).\n")

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    total_tp = sum(r["tp"] for r in per_image_rows)
    total_fp = sum(r["fp"] for r in per_image_rows)
    total_fn = sum(r["fn"] for r in per_image_rows)
    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1   = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)
                if (agg_prec + agg_rec) > 0 else 0.0)

    iou_thresholds_5095 = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]
    ap_dict = compute_map_across_thresholds(
        all_preds_per_image, all_gt_per_image, iou_thresholds_5095
    )
    map50   = ap_dict[0.5]
    map5095 = float(np.mean(list(ap_dict.values())))

    # Inference speed (mean latency and FPS over evaluated images)
    mean_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    inference_fps = 1000.0 / mean_inference_ms if mean_inference_ms > 0 else 0.0

    summary = {
        "model": model_type,
        "weights": weights_path,
        "conf_threshold": conf_threshold,
        "num_images_evaluated": len(per_image_rows),
        "num_images_skipped": skipped,
        "total_gt_boxes": total_gt_05,
        "total_pred_boxes": sum(r["num_pred"] for r in per_image_rows),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision":      round(agg_prec,  4),
        "recall":         round(agg_rec,   4),
        "f1":             round(agg_f1,    4),
        "mAP@0.5":        round(map50,     4),
        "mAP@0.5:0.95":   round(map5095,   4),
        "AP_per_IoU_threshold": {str(k): round(v, 4) for k, v in ap_dict.items()},
        "inference_mean_ms": round(mean_inference_ms, 2),
        "inference_fps":     round(inference_fps, 2),
    }

    # -----------------------------------------------------------------------
    # PR / F1 curve data
    # -----------------------------------------------------------------------
    confs, precisions, recalls = build_pr_points(all_matches_05, total_gt_05)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    # 1. metrics_summary.json
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 2. per_image_results.csv
    per_image_path = os.path.join(output_dir, "per_image_results.csv")
    fieldnames = ["file_name", "num_gt", "num_pred", "tp", "fp", "fn", "precision", "recall", "inference_ms"]
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    # 3. predictions_detailed.csv
    detailed_path = os.path.join(output_dir, "predictions_detailed.csv")
    save_predictions_csv(all_detailed, detailed_path)

    # 4. PR curve
    pr_path = os.path.join(output_dir, "pr_curve.png")
    if len(precisions) > 0:
        plot_pr_curve(precisions, recalls, map50, model_type, pr_path)
    else:
        print("  Warning: no predictions to plot PR curve.")

    # 5. F1-confidence curve
    f1_path = os.path.join(output_dir, "f1_curve.png")
    if len(confs) > 0:
        plot_f1_curve(confs, precisions, recalls, model_type, f1_path)
    else:
        print("  Warning: no predictions to plot F1 curve.")

    # 6. Confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(total_tp, total_fp, total_fn, model_type, cm_path, normalize=True)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("=" * 52)
    print(f"  Model          : {model_type}")
    print(f"  Images         : {len(per_image_rows)}")
    print(f"  Total GT boxes : {total_gt_05}")
    print(f"  Total predicted: {summary['total_pred_boxes']}")
    print(f"  TP / FP / FN   : {total_tp} / {total_fp} / {total_fn}")
    print(f"  Precision      : {summary['precision']}")
    print(f"  Recall         : {summary['recall']}")
    print(f"  F1             : {summary['f1']}")
    print(f"  mAP@0.5        : {summary['mAP@0.5']}")
    print(f"  mAP@0.5:0.95   : {summary['mAP@0.5:0.95']}")
    print(f"  Inference      : {summary['inference_mean_ms']} ms/img  ({summary['inference_fps']} FPS)")
    print("=" * 52)
    print(f"\nResults saved to: {output_dir}")
    for p in [summary_path, per_image_path, detailed_path, pr_path, f1_path, cm_path]:
        print(f"  {p}")


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
        choices=["yolov11", "yolov26", "rfdetr"],
        help="Model architecture: 'yolov11', 'yolov26', or 'rfdetr'",
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
