#!/usr/bin/env python3
"""Verify the frozen test image/label manifest without modifying it."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "ml" / "data"
MANIFEST = DATA_ROOT / "test_manifest.sha256"
TEST_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_test_manifest(
    manifest_path: Path = MANIFEST,
    data_root: Path = DATA_ROOT,
) -> list[str]:
    if not manifest_path.is_file():
        return [f"Frozen test manifest is missing: {manifest_path}"]
    expected: dict[str, str] = {}
    errors = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"line {line_number}: expected '<sha256>  <relative path>'")
            continue
        digest, relative = parts[0].lower(), parts[1].strip()
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"line {line_number}: path must stay inside the data root")
            continue
        if relative in expected:
            errors.append(f"line {line_number}: duplicate path {relative}")
            continue
        if not (
            relative.startswith("merged/images/test/")
            or relative.startswith("merged/labels/test/")
        ):
            errors.append(f"line {line_number}: path is outside frozen test images/labels: {relative}")
            continue
        expected[relative] = digest

    if not expected:
        errors.append("manifest contains no frozen test files")
        return errors

    expected_paths = set(expected)
    actual_paths = set()
    for directory, suffixes in (
        (data_root / "merged" / "images" / "test", TEST_IMAGE_SUFFIXES),
        (data_root / "merged" / "labels" / "test", {".txt"}),
    ):
        if not directory.is_dir():
            errors.append(f"frozen test directory is missing: {directory}")
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                actual_paths.add(path.relative_to(data_root).as_posix())

    for relative in sorted(expected_paths - actual_paths):
        errors.append(f"manifest file is missing: {relative}")
    for relative in sorted(actual_paths - expected_paths):
        errors.append(f"unmanifested frozen test file: {relative}")
    for relative in sorted(expected_paths & actual_paths):
        path = data_root / relative
        if file_sha256(path) != expected[relative]:
            errors.append(f"SHA256 changed: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    errors = verify_test_manifest(args.manifest, args.data_root)
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print("Frozen test images and labels match the checked-in manifest.")


if __name__ == "__main__":
    main()
