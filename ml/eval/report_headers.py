"""Build and verify immutable provenance headers for evaluation reports."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[2]
CLASS_FILE = ROOT / "ml" / "configs" / "classes.yaml"
DATASET_HASHES = ("source_hash_manifest", "test_manifest", "dataset_yaml", "video_labels")
METRIC_DEFINITIONS = {
    "true_positive": "One-to-one greedy confidence-ranked match of same class at IoU >= 0.50.",
    "image_ap": "Interpolated precision-recall area; GT-free classes are NOT EVALUABLE, not zero AP.",
    "incident_recall": "True detected labeled accident events divided by all labeled test events; exact two-sided Clopper-Pearson 95% interval over events.",
    "false_alarms": "False incident events divided by accident-free camera-hours; zero false alarms report exact one-sided 95% Poisson upper bound, approximately 2.996/T per hour.",
    "bootstrap": "Resample camera/scene clusters, never frames; report percentile 95% interval.",
    "test_protocol": "Tune thresholds only with rows labeled tuning; inspect the frozen test split once.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_manifest_hash(data_manifest_path: Path) -> str:
    data_manifest_path = data_manifest_path.resolve()
    manifest = yaml.safe_load(data_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Evaluation data manifest must be a YAML mapping.")
    digest = hashlib.sha256()
    paths = [data_manifest_path]
    for key in DATASET_HASHES:
        value = manifest.get(key)
        if value:
            path = ROOT / value
            paths.append(path)
    for path in sorted(paths):
        try:
            identity = path.relative_to(ROOT).as_posix()
        except ValueError:
            identity = path.resolve().as_posix()
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"MISSING")
        digest.update(b"\0")
    return digest.hexdigest()


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_header(model_path: Path, data_manifest_path: Path) -> dict[str, Any]:
    return {
        "run_id": str(uuid4()),
        "git_commit": current_git_commit(),
        "model_sha256": sha256_file(model_path),
        "data_manifest_sha256": data_manifest_hash(data_manifest_path),
        "class_file_sha256": sha256_file(CLASS_FILE),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "metric_definitions": METRIC_DEFINITIONS,
    }


def format_header(header: dict[str, Any], scope: dict[str, Any]) -> str:
    document = {**header, "video_scope": scope}
    return "---\n" + yaml.safe_dump(document, sort_keys=False, allow_unicode=True) + "---\n\n"


def parse_report_header(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Report is missing provenance front matter: {report_path}")
    _, front_matter, _ = text.split("---\n", 2)
    header = yaml.safe_load(front_matter)
    if not isinstance(header, dict):
        raise ValueError(f"Malformed report provenance header: {report_path}")
    return header
