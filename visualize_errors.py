"""
Error Visualization Module - ALPR Object Detection System

Reads `predictions_detailed.csv` produced by `evaluate.py`, finds detections with
IoU < 0.5 (including false positives, which already have IoU < 0.5 by definition),
and overlays predicted bounding boxes and confidence scores on the original images.

For each model, results are written to a separate directory:

    runs/qualitative/{model}/iou_lt_0_5/

Usage:
    python visualize_errors.py \
        --model yolov11 \
        --images path/to/test/images \
        --predictions runs/evaluation/yolov11/predictions_detailed.csv

    python visualize_errors.py \
        --model yolov10 \
        --images path/to/test/images \
        --predictions runs/evaluation/yolov10/predictions_detailed.csv

    python visualize_errors.py \
        --model rtdetrv2 \
        --images path/to/test/images \
        --predictions runs/evaluation/rtdetrv2/predictions_detailed.csv

Options:
    --output  Custom output directory (default: runs/qualitative/{model}/iou_lt_0_5)
    --limit   Max number of images to save (per run), for quick inspection

Only detections with a predicted bounding box are visualized (TP and FP rows).
FN rows have no predicted box and are therefore not drawn.

Author: ALPR Thesis Project
"""

import argparse
import os
import csv
from collections import defaultdict

import cv2


def load_error_cases(predictions_csv, iou_threshold=0.5):
    """
    Load detections from predictions_detailed.csv and return all rows
    with IoU < iou_threshold that also have a predicted box (TP and FP).

    Args:
        predictions_csv (str): Path to predictions_detailed.csv.
        iou_threshold (float): IoU cutoff for considering a case an "error".

    Returns:
        dict: {file_name: [record_dict, ...], ...}
              where record_dict contains keys:
                  file_name, match_type, pred_xmin, pred_ymin,
                  pred_xmax, pred_ymax, confidence, iou
    """
    error_cases = defaultdict(list)

    with open(predictions_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_type = row.get("match_type", "").strip()
            # Only TP and FP have predicted boxes; FNs have them empty
            if match_type not in {"TP", "FP"}:
                continue

            # Some rows may have empty IoU / confidence if malformed; skip them
            try:
                iou = float(row.get("iou", "0") or 0.0)
            except ValueError:
                iou = 0.0

            if iou >= iou_threshold:
                continue

            try:
                pxmin = float(row.get("pred_xmin", ""))
                pymin = float(row.get("pred_ymin", ""))
                pxmax = float(row.get("pred_xmax", ""))
                pymax = float(row.get("pred_ymax", ""))
            except ValueError:
                # Skip records without valid predicted coordinates
                continue

            try:
                conf = float(row.get("confidence", ""))
            except ValueError:
                conf = 0.0

            filename = row["file_name"]
            error_cases[filename].append(
                {
                    "match_type": match_type,
                    "pred_box": [pxmin, pymin, pxmax, pymax],
                    "confidence": conf,
                    "iou": iou,
                }
            )

    return error_cases


def draw_errors_on_image(img, error_records):
    """
    Draw all error detections on the given image.

    - Red boxes for FP
    - Orange boxes for TP with IoU < threshold (i.e., misaligned)
    - Text: confidence and IoU above each box

    Args:
        img (np.ndarray): BGR image (OpenCV).
        error_records (list): List of dicts from load_error_cases().

    Returns:
        np.ndarray: Annotated image (BGR).
    """
    annotated = img.copy()

    for rec in error_records:
        pxmin, pymin, pxmax, pymax = rec["pred_box"]
        conf = rec["confidence"]
        iou = rec["iou"]
        match_type = rec["match_type"]

        # Choose color based on error type
        if match_type == "FP":
            color = (0, 0, 255)  # Red for false positives
            label = f"FP conf={conf:.2f} IoU={iou:.2f}"
        else:
            color = (0, 165, 255)  # Orange for low-IoU true positives
            label = f"TP (IoU<{0.5:.2f}) conf={conf:.2f} IoU={iou:.2f}"

        pt1 = (int(pxmin), int(pymin))
        pt2 = (int(pxmax), int(pymax))

        cv2.rectangle(annotated, pt1, pt2, color, thickness=2)

        # Text background
        text_scale = 0.5
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        text_origin = (pt1[0], max(0, pt1[1] - 5))
        text_bottom_left = (text_origin[0], text_origin[1] - text_h - baseline)
        text_top_right = (text_origin[0] + text_w, text_origin[1] + baseline)

        cv2.rectangle(
            annotated,
            text_bottom_left,
            text_top_right,
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (text_origin[0], text_origin[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def visualize_errors(model_type, images_dir, predictions_csv, output_dir, limit=None):
    """
    Visualize all detections with IoU < 0.5 by drawing predicted boxes and
    confidence scores onto the corresponding images and saving them.

    Args:
        model_type (str): 'yolov11', 'yolov10', or 'rtdetrv2' (for output organisation only).
        images_dir (str): Directory containing the original test images.
        predictions_csv (str): Path to predictions_detailed.csv from evaluate.py.
        output_dir (str): Directory to save annotated error images.
        limit (int or None): Max number of images to save (for quick inspection).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading error cases from: {predictions_csv}")
    error_cases = load_error_cases(predictions_csv, iou_threshold=0.5)

    if not error_cases:
        print("No IoU < 0.5 error cases found. Nothing to visualize.")
        return

    print(f"Found {len(error_cases)} images with IoU < 0.5 cases.")

    saved = 0
    for filename, records in error_cases.items():
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: could not load image: {img_path}")
            continue

        annotated = draw_errors_on_image(img, records)

        save_name = os.path.splitext(filename)[0] + "_errors.jpg"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, annotated)
        saved += 1

        if limit is not None and saved >= limit:
            print(f"Reached limit of {limit} images. Stopping.")
            break

    print(f"Saved {saved} annotated error images to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize low-IoU detections (IoU < 0.5) using predictions_detailed.csv."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov11", "yolov10", "rtdetrv2"],
        help="Model architecture (used for organising output directories).",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to directory containing the original test images.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions_detailed.csv produced by evaluate.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save annotated images. "
             "Default: runs/qualitative/{model}/iou_lt_0_5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to save (default: no limit).",
    )

    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        "runs", "qualitative", args.model, "iou_lt_0_5"
    )

    visualize_errors(
        model_type=args.model,
        images_dir=args.images,
        predictions_csv=args.predictions,
        output_dir=output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

