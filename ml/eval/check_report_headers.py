#!/usr/bin/env python3
"""Fail when the latest run report hashes do not match current evaluation inputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.eval.report_headers import (
    CLASS_FILE,
    data_manifest_hash,
    parse_report_header,
    sha256_file,
)


def verify_latest(
    model_path: Path,
    data_manifest_path: Path,
    reports_root: Path = ROOT / "ml" / "reports",
) -> list[str]:
    pointer = reports_root / "latest_run.txt"
    if not pointer.is_file():
        return [f"Latest evaluation pointer is missing: {pointer}"]
    run_id = pointer.read_text(encoding="utf-8").strip()
    if not run_id or Path(run_id).name != run_id:
        return ["Latest evaluation pointer contains an invalid run id."]
    run_dir = reports_root / "runs" / run_id
    errors = []
    expected = {
        "model_sha256": sha256_file(model_path),
        "data_manifest_sha256": data_manifest_hash(data_manifest_path),
        "class_file_sha256": sha256_file(CLASS_FILE),
    }
    report_paths = sorted(run_dir.glob("*.md"))
    if not report_paths:
        return [f"Latest evaluation run has no reports: {run_dir}"]
    for report_path in report_paths:
        try:
            header = parse_report_header(report_path)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        if header.get("run_id") != run_id:
            errors.append(f"{report_path}: run_id does not match latest_run.txt")
        for field, value in expected.items():
            if header.get(field) != value:
                errors.append(f"{report_path}: {field} does not match current input")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--data-manifest", type=Path, default=ROOT / "ml" / "eval" / "data_manifest.yaml")
    args = parser.parse_args()
    errors = verify_latest(args.model, args.data_manifest)
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("Latest evaluation report hashes match current model, data manifest, and taxonomy.")


if __name__ == "__main__":
    main()
