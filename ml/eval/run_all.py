#!/usr/bin/env python3
"""Generate same-run evaluation artifacts and fail closed on missing evidence."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.eval.diagnose import evaluate_model
from ml.eval.report_headers import make_header, format_header


REPORTS = (
    "per_class_metrics.md",
    "slice_analysis.md",
    "robustness.md",
    "threshold_tuning.md",
    "temporal_model_evaluation.md",
    "full_pipeline.md",
)


def _resolve(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def video_scope(labels_path: Path) -> dict[str, Any]:
    scope = {
        "test_clips": 0,
        "test_events": 0,
        "test_cameras": 0,
        "test_scenes": 0,
        "test_accident_cameras": 0,
        "test_accident_scenes": 0,
        "test_camera_hours": 0.0,
        "test_accident_free_camera_hours": 0.0,
        "tuning_clips": 0,
    }
    if not labels_path.is_file():
        return scope
    test_cameras: set[str] = set()
    test_scenes: set[str] = set()
    accident_cameras: set[str] = set()
    accident_scenes: set[str] = set()
    with labels_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("split") == "tuning":
                scope["tuning_clips"] += 1
            if row.get("split") != "test":
                continue
            scope["test_clips"] += 1
            test_cameras.add(row.get("camera_id", ""))
            test_scenes.add(row.get("scene_id", ""))
            duration = float(row["duration_seconds"])
            scope["test_camera_hours"] += duration / 3600.0
            if (row.get("contains_accident") or "").lower() == "true":
                scope["test_events"] += 1
                accident_cameras.add(row.get("camera_id", ""))
                accident_scenes.add(row.get("scene_id", ""))
            else:
                scope["test_accident_free_camera_hours"] += duration / 3600.0
    scope["test_cameras"] = len(test_cameras - {""})
    scope["test_scenes"] = len(test_scenes - {""})
    scope["test_accident_cameras"] = len(accident_cameras - {""})
    scope["test_accident_scenes"] = len(accident_scenes - {""})
    return scope


def generate_run(model_path: Path, manifest_path: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Data manifest must be a YAML mapping: {manifest_path}")
    dataset_yaml = _resolve(manifest["dataset_yaml"], manifest_path)
    labels_path = _resolve(manifest["video_labels"], manifest_path)
    scope = video_scope(labels_path)
    header = make_header(model_path, manifest_path)
    run_dir = output_root / header["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_ready = dataset_yaml.is_file()
    video_labels_ready = labels_path.is_file() and scope["test_clips"] > 0
    status = "BLOCKED"
    reasons = []
    if not dataset_ready:
        reasons.append(f"rebuilt image dataset is missing: {dataset_yaml}")
    if not video_labels_ready:
        reasons.append(f"validated tuning/test video labels are missing: {labels_path}")
    reasons.append(
        "full tracking/debounce/temporal/VLM incident metrics are not implemented in this "
        "shared runner; legacy reports are not imported"
    )

    report_status = {}
    if dataset_ready:
        temp_metrics = run_dir / "per_class_metrics.body.md"
        try:
            evaluate_model(model_path, dataset_yaml, temp_metrics)
            body = temp_metrics.read_text(encoding="utf-8")
            temp_metrics.unlink()
            report_status["per_class_metrics.md"] = "COMPUTED"
        except (ImportError, FileNotFoundError, ValueError) as error:
            body = f"# Per-class metrics — PRELIMINARY\n\nBLOCKED: {error}\n"
            report_status["per_class_metrics.md"] = f"BLOCKED: {error}"
    else:
        body = (
            "# Per-class metrics — PRELIMINARY / BLOCKED\n\n"
            "No per-class AP, GT boxes, or prediction counts are reported because the rebuilt "
            "test split does not exist.\n"
        )
        report_status["per_class_metrics.md"] = "BLOCKED: rebuilt test split missing"
    (run_dir / "per_class_metrics.md").write_text(
        format_header(header, scope) + body,
        encoding="utf-8",
    )

    unavailable_reasons = reasons or [
        "the shared video metric implementation requires reviewed event annotations and "
        "source-verified clip metadata; no legacy report is imported into this run"
    ]
    for filename in REPORTS[1:]:
        title = filename.removesuffix(".md").replace("_", " ").title()
        body = (
            f"# {title} — PRELIMINARY / {status}\n\n"
            + "\n".join(f"- {reason}" for reason in unavailable_reasons)
            + "\n\nNo metrics are inferred from historical reports or plots.\n"
        )
        (run_dir / filename).write_text(format_header(header, scope) + body, encoding="utf-8")
        report_status[filename] = status

    artifact = {
        "run_header": header,
        "video_scope": scope,
        "report_status": report_status,
        "blocked_reasons": reasons,
        "leakage_failures": None,
        "unverified_claims": None,
        "metrics": None,
        "promotion_ready": False,
    }
    (run_dir / "run_artifacts.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root.parent / "latest_run.txt").write_text(header["run_id"] + "\n", encoding="utf-8")
    return run_dir, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=ROOT / "ml" / "eval" / "data_manifest.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ml" / "reports" / "runs")
    args = parser.parse_args()
    run_dir, artifact = generate_run(args.model, args.data_manifest, args.output_dir)
    print(f"Evaluation run: {run_dir}")
    if artifact["blocked_reasons"]:
        raise SystemExit("Evaluation blocked by missing evidence; see the run artifacts.")


if __name__ == "__main__":
    main()
