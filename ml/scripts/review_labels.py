#!/usr/bin/env python3
"""Rank likely accident-label disagreements and export human-review contact sheets."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acs.taxonomy import class_names, decode_model_class, load_classes
from ml.eval.metrics import box_iou

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _xywh_to_xyxy(box: tuple[float, float, float, float], width: int, height: int):
    xc, yc, box_width, box_height = box
    return (
        (xc - box_width / 2) * width,
        (yc - box_height / 2) * height,
        (xc + box_width / 2) * width,
        (yc + box_height / 2) * height,
    )


def _read_labels(label_path: Path, names: list[str], width: int, height: int):
    boxes = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label at {label_path}:{line_number}")
        class_id = int(parts[0])
        if not 0 <= class_id < len(names):
            raise ValueError(f"Unknown canonical class {class_id} at {label_path}:{line_number}")
        boxes.append((class_id, _xywh_to_xyxy(tuple(map(float, parts[1:])), width, height)))
    return boxes


def rank_image(
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    accident_ids: set[int],
    confidence_floor: float,
) -> tuple[float, str] | None:
    accident_gt = [(class_id, box) for class_id, box in ground_truth if class_id in accident_ids]
    accident_predictions = [
        item for item in predictions if item[0] in accident_ids and item[1] >= confidence_floor
    ]
    if accident_gt:
        worst_quality = 1.0
        worst_other_overlap = 0.0
        for class_id, gt_box in accident_gt:
            matching = [
                (confidence * box_iou(gt_box, pred_box), confidence, pred_box)
                for pred_class, confidence, pred_box in predictions
                if pred_class == class_id
            ]
            best_quality = max((quality for quality, _, _ in matching), default=0.0)
            worst_quality = min(worst_quality, best_quality)
            other_overlap = max(
                (
                    confidence * box_iou(gt_box, pred_box)
                    for pred_class, confidence, pred_box in predictions
                    if pred_class not in accident_ids
                ),
                default=0.0,
            )
            worst_other_overlap = max(worst_other_overlap, other_overlap)
        if worst_quality < 0.50:
            if worst_other_overlap >= 0.50:
                return worst_other_overlap, "confident_other_class_overlaps_accident_label"
            return 1.0 - worst_quality, "accident_ground_truth_not_supported_by_prediction"
    elif accident_predictions:
        top = max(accident_predictions, key=lambda item: item[1])
        return top[1], "possible_unlabeled_accident_prediction"
    elif not ground_truth:
        possible_objects = [
            item for item in predictions if item[1] >= confidence_floor
        ]
        if possible_objects:
            top = max(possible_objects, key=lambda item: item[1])
            return top[1], "possible_unlabeled_vehicle_or_road_user"
    return None


def export_review(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    limit_per_class: int = 200,
    confidence_floor: float = 0.50,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image split is missing: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label split is missing: {labels_dir}")
    model = YOLO(str(model_path))
    canonical = load_classes()
    names = class_names()
    accident_ids = {item["id"] for item in canonical if item["group"] == "accident"}
    candidates: dict[int, list[dict[str, Any]]] = {class_id: [] for class_id in accident_ids}
    all_images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in all_images:
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth = _read_labels(
            labels_dir / f"{image_path.stem}.txt", names, width, height
        )
        result = model.predict(str(image_path), conf=0.01, device=device, verbose=False)[0]
        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                model_id = int(box.cls.item())
                canonical_id, _, _ = decode_model_class(model_id, model.names)
                predictions.append(
                    (
                        canonical_id,
                        float(box.conf.item()),
                        tuple(float(value) for value in box.xyxy[0].tolist()),
                    )
                )
        ranked = rank_image(ground_truth, predictions, accident_ids, confidence_floor)
        if ranked is None:
            continue
        score, reason = ranked
        for class_id in accident_ids:
            candidates[class_id].append(
                {
                    "image": image_path,
                    "score": score,
                    "reason": reason,
                    "gt_accident_boxes": sum(item[0] == class_id for item in ground_truth),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    csv_path = output_dir / "suspicious_images.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("class_id", "class_name", "image", "suspicion_score", "reason", "gt_boxes"),
        )
        writer.writeheader()
        for class_id, items in candidates.items():
            selected = sorted(items, key=lambda item: (-item["score"], str(item["image"])))[:limit_per_class]
            _write_contact_sheet(selected, output_dir / f"accident_class_{class_id}_review.jpg")
            for item in selected:
                writer.writerow(
                    {
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "image": item["image"].relative_to(images_dir).as_posix(),
                        "suspicion_score": f"{item['score']:.6f}",
                        "reason": item["reason"],
                        "gt_boxes": item["gt_accident_boxes"],
                    }
                )
            results[names[class_id]] = len(selected)

    summary = {
        "status": "REVIEW_REQUIRED",
        "candidate_images_per_accident_class": results,
        "reviewed_images": 0,
        "confirmed_label_errors": None,
        "label_error_rate_before": None,
        "label_error_rate_after": None,
        "note": (
            "Model disagreement ranks candidates; it is not a confirmed label error. "
            "Error rates require human keep/fix/drop decisions and applied annotation corrections."
        ),
        "decisions_file": str(ROOT / "ml" / "configs" / "class_map.yaml"),
    }
    (output_dir / "label_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_contact_sheet(items: list[dict[str, Any]], output_path: Path) -> None:
    if not items:
        return
    columns, tile_width, tile_height = 4, 320, 230
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB")
        image.thumbnail((tile_width - 8, tile_height - 38))
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        sheet.paste(image, (x + 4, y + 4))
        label = f"{item['score']:.3f} {item['reason']}"
        draw.text((x + 4, y + tile_height - 28), label[:48], fill="black")
        draw.text((x + 4, y + tile_height - 14), item["image"].name[:48], fill="black")
    sheet.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--images", type=Path, default=ROOT / "ml/data/merged/images/train")
    parser.add_argument("--labels", type=Path, default=ROOT / "ml/data/merged/labels/train")
    parser.add_argument("--output", type=Path, default=ROOT / "ml/reports/label_review")
    parser.add_argument("--limit-per-class", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = export_review(
        args.model,
        args.images,
        args.labels,
        args.output,
        args.limit_per_class,
        args.confidence,
        args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
