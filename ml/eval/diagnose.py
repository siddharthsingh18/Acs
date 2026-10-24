#!/usr/bin/env python3
"""Evaluate current weights against a rebuilt canonical test split without retraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acs.taxonomy import class_names, decode_model_class, load_classes
from ml.eval.metrics import class_metrics


def _data_yaml_parts(data_yaml: Path) -> tuple[Path, Path, list[str]]:
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = document.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=int)]
    if names != class_names():
        raise ValueError(
            f"Test manifest class order {names!r} differs from canonical classes {class_names()!r}."
        )
    base = Path(document.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    return base.resolve(), Path(document["test"]), names


def evaluate_model(model_path: Path, data_yaml: Path, report_path: Path, confidence: float = 0.001) -> None:
    from PIL import Image
    from ultralytics import YOLO

    base, test_relative, names = _data_yaml_parts(data_yaml)
    image_dir = (base / test_relative).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    label_dir = base / "labels" / "test"
    model = YOLO(str(model_path))
    model_names = model.names
    gt_by_class = {name: 0 for name in names}
    ground_truth = []
    predictions = []
    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                tokens = line.split()
                if len(tokens) != 5:
                    raise ValueError(f"Malformed label at {label_path}:{line_number}")
                class_id, xc, yc, width, height = int(tokens[0]), *map(float, tokens[1:])
                if not 0 <= class_id < len(names):
                    raise ValueError(f"Out-of-taxonomy GT class ID {class_id} in {label_path}")
                box = (
                    (xc - width / 2) * image_width,
                    (yc - height / 2) * image_height,
                    (xc + width / 2) * image_width,
                    (yc + height / 2) * image_height,
                )
                ground_truth.append((str(image_path), class_id, box))
                gt_by_class[names[class_id]] += 1

        result = model.predict(source=str(image_path), conf=confidence, verbose=False)[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            legacy_id = int(box.cls.item())
            canonical_id, _, _ = decode_model_class(legacy_id, model_names)
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            predictions.append(
                (str(image_path), canonical_id, float(box.conf.item()), (x1, y1, x2, y2))
            )

    class_ids = list(range(len(names)))
    metrics = class_metrics(ground_truth, predictions, class_ids, iou_threshold=0.50)
    accident_ids = {
        item["id"] for item in load_classes() if item["group"] == "accident"
    }
    rows = [
        "# Current checkpoint diagnosis",
        "",
        "> **PRELIMINARY / BLOCKED:** This is a diagnostic, not a promotion evaluation. Do not train or tune until labels are reviewed and this report is regenerated on the frozen test set.",
        "",
        f"- Model: `{model_path}`",
        f"- Test data manifest: `{data_yaml}`",
        f"- Test images: {len(image_paths)}",
        f"- Prediction confidence floor: {confidence}",
        "- AP definition: confidence-ranked greedy one-to-one matching at IoU ≥ 0.50; AP is area under the interpolated precision-recall curve.",
        "",
        "| Class | AP@0.50 | GT boxes | Predictions |",
        "|---|---:|---:|---:|",
    ]
    for class_id, name in enumerate(names):
        item = metrics[class_id]
        not_evaluable = item["ap"] is None or (
            class_id in accident_ids and int(item["ground_truth"]) < 30
        )
        ap = "NOT EVALUABLE" if not_evaluable else f"{float(item['ap']):.6f}"
        rows.append(
            f"| {name} | {ap} | {item['ground_truth']} | {item['predictions']} |"
        )
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_blocked_diagnosis(report_path: Path) -> None:
    report = """# Current checkpoint diagnosis — PRELIMINARY / BLOCKED

**Training is prohibited until this diagnosis can be regenerated on a new leakage-safe test split and label review is complete.**

## Evidence available in this checkout

- The canonical detector taxonomy is four classes: `bike`, `car`, `person`, `accident` (`ml/configs/classes.yaml`).
- The checked-in YOLO checkpoint is a legacy 10-output model: `bike`, `bike_bike_accident`, `bike_object_accident`, `bike_person_accident`, `car`, `car_bike_accident`, `car_car_accident`, `car_object_accident`, `car_person_accident`, `person` (`ml/configs/checkpoint_class_mappings.yaml`).
- The checkpoint class order differs from the canonical class order. Existing inference has an explicit decoder, but class-index-based metrics from `model.val()` are invalid unless the checkpoint is canonical; the evaluator now refuses that silent mismatch.
- `ml/data/` contains no images, labels, `data.yaml`, provenance file, or frozen test manifest. No old split is reused.
- Therefore current per-class AP, predictions, and GT box counts cannot be computed. **Unavailable is not zero.**

## Current-weight evaluation on the new test split

| Class | AP@0.50 | GT boxes | Predictions | Verdict |
|---|---:|---:|---:|---|
| bike | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| car | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| person | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |
| accident | NOT COMPUTABLE | NOT AVAILABLE | NOT AVAILABLE | No rebuilt test set |

The runnable evaluator is `ml/eval/diagnose.py`. It decodes model IDs explicitly into canonical IDs, counts all predictions above its documented confidence floor, and reports AP only when labels and test images exist.

## Accident label coverage

| Split / phase | Accident GT boxes | Finding |
|---|---:|---|
| Previous train / val / test | NOT AVAILABLE | No old split or dataset is present; historical source-wide counts are not training-split counts. |
| Rebuilt train / val / test | NOT AVAILABLE | No approved source archive or provenance metadata is available; the rebuild has not run. |

## Root causes and fixes

1. **Taxonomy/index mismatch is confirmed.** The legacy detector emits 10 IDs while the current data taxonomy is four classes. Runtime inference must decode through the explicit mapping; raw index-based test metrics can report the wrong class or no GT for accident labels. The mapping fixes interpretation only; it does not retrain or improve the weights.
2. **Accident label coverage is not established.** The previous model card says most accident classes lacked test GT, but there are no split label files to verify those counts. A new grouped split must report counts; any accident class below 30 GT boxes in val or test is marked NOT EVALUABLE and blocks promotion.
3. **Label quality and leakage are unknown.** Images, source provenance, and camera/scene identities are missing. Perceptual-hash plus local CLIP/DINOv2 clustering, label review, and group-level splitting must run before evaluation.

## Required next action

Clear source licenses and archive checksums, fetch the source data, add complete camera/scene/video provenance, provide a local CLIP/DINOv2 TorchScript encoder, and review the exported contact sheets. Then rebuild the dataset, freeze its test manifest, and run `python -m ml.eval.diagnose --model best.pt --data-yaml ml/data/merged/data.yaml`. No retraining or test threshold tuning is authorized before this report has real GT counts and per-class AP.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "ml" / "data" / "merged" / "data.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "ml" / "reports" / "diagnosis.md")
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()
    if not args.data_yaml.is_file():
        write_blocked_diagnosis(args.report)
        raise SystemExit(f"No rebuilt data manifest; diagnostic blocked. See {args.report}.")
    evaluate_model(args.model, args.data_yaml, args.report, args.confidence)


if __name__ == "__main__":
    main()
