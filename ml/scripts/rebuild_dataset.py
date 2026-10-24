#!/usr/bin/env python3
"""Build a duplicate-safe YOLO dataset from approved, provenance-complete source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image
from scipy.fftpack import dct

from acs.taxonomy import class_names, load_classes
from ml.scripts.fetch_data import sha256_file


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "ml" / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROVENANCE_PATH = DATA_ROOT / "provenance.csv"
CLASS_MAP_PATH = ROOT / "ml" / "configs" / "class_map.yaml"
OUTPUT_ROOT = DATA_ROOT / "merged"
TEST_MANIFEST_PATH = DATA_ROOT / "test_manifest.sha256"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROVENANCE_FIELDS = {"relative_path", "source_id", "video_id", "camera_id", "scene_id"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_ACCIDENT_BOXES = 30
MIN_BOX_SIDE_PIXELS = 4


@dataclass
class ImageRecord:
    path: Path
    relative_path: str
    label_path: Path
    source_id: str
    group_id: str
    scene_id: str
    boxes: list[tuple[int, float, float, float, float]]
    phash: np.ndarray
    embedding: np.ndarray | None = None
    split: str | None = None
    review_reasons: list[str] | None = None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def perceptual_hash(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    low = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    median = np.median(low[1:, 1:])
    return low > median


def read_provenance(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Add source IDs and camera/scene/video groups; "
            "filename-derived groups are not accepted."
        )
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(row for row in stream if not row.lstrip().startswith("#"))
        if not reader.fieldnames or not PROVENANCE_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"Provenance CSV must include: {', '.join(sorted(PROVENANCE_FIELDS))}")
        for row in reader:
            relative_path = (row.get("relative_path") or "").strip()
            if not relative_path or relative_path in rows:
                raise ValueError(f"Missing or duplicate provenance path: {relative_path!r}")
            if not (row.get("source_id") or "").strip():
                raise ValueError(f"Missing source_id for {relative_path}")
            group = (row.get("video_id") or "").strip() or (row.get("camera_id") or "").strip() or (
                row.get("scene_id") or ""
            ).strip()
            if not group:
                raise ValueError(f"No video_id, camera_id, or scene_id group for {relative_path}")
            rows[relative_path] = {
                "source_id": row["source_id"].strip(),
                "group_id": f"group:{group}",
                "scene_id": (row.get("scene_id") or "").strip(),
                "camera_id": (row.get("camera_id") or "").strip(),
            }
    return rows


def load_source_mappings(path: Path = CLASS_MAP_PATH) -> dict[str, dict[int, dict]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    mappings = {}
    for source_id, source in (document.get("datasets") or {}).items():
        source_mappings = {}
        for source_class, mapping in (source.get("class_mapping") or {}).items():
            source_mappings[int(source_class)] = mapping
        mappings[source_id] = source_mappings
    return mappings


def load_review_decisions(path: Path = CLASS_MAP_PATH) -> dict[tuple[str, int, int], dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = class_names()
    decisions = {}
    for entry in document.get("label_review_decisions") or []:
        image = str(entry.get("image", "")).strip()
        source_id = str(entry.get("source_id", "")).strip()
        line_number = entry.get("line_number")
        source_class_id = entry.get("source_class_id")
        decision = entry.get("decision")
        target_class = entry.get("target_class")
        reason = str(entry.get("reason", "")).strip()
        image_path = Path(image)
        if (
            not image
            or image_path.is_absolute()
            or ".." in image_path.parts
            or not source_id
            or not reason
        ):
            raise ValueError("Each label review decision needs image, source_id, and reason.")
        if not isinstance(line_number, int) or line_number < 1:
            raise ValueError(f"Invalid line_number in label review decision for {image}.")
        if not isinstance(source_class_id, int) or source_class_id < 0:
            raise ValueError(f"Invalid source_class_id in label review decision for {image}.")
        if decision not in {"keep", "fix", "drop"}:
            raise ValueError(f"Decision for {image} must be keep, fix, or drop.")
        if decision == "drop":
            if target_class not in (None, ""):
                raise ValueError(f"Dropped label for {image} must not specify target_class.")
        elif target_class not in names:
            raise ValueError(f"Reviewed label for {image} must specify a canonical target_class.")
        key = (image, line_number, source_class_id)
        if key in decisions:
            raise ValueError(f"Duplicate label review decision for {image}:{line_number}.")
        decisions[key] = {
            "decision": decision,
            "target_class": target_class,
            "reason": reason,
            "source_id": source_id,
        }
    return decisions


def require_approved_sources() -> set[str]:
    manifest_path = ROOT / "ml" / "configs" / "data_sources.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    approved = {
        source["id"]
        for source in manifest.get("sources", [])
        if source.get("enabled") is True
        and source.get("status") == "approved"
        and source.get("redistribution_allowed") is True
        and source.get("model_training_allowed") is True
    }
    if not approved:
        raise ValueError(
            "No sources are licensed and approved for training. Resolve ml/configs/data_sources.yaml "
            "before rebuilding; do not use unverified source files."
        )
    return approved


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def clean_yolo_label(
    label_path: Path,
    mapping: dict[int, dict],
    image_width: int,
    image_height: int,
    image_relative_path: str | None = None,
    review_decisions: dict[tuple[str, int, int], dict] | None = None,
    used_review_decisions: set[tuple[str, int, int]] | None = None,
    dataset_source_id: str | None = None,
) -> tuple[list[tuple[int, float, float, float, float]], list[str], bool]:
    if not label_path.is_file():
        return [], ["missing_label_file"], True
    boxes = []
    reasons = []
    unresolved = False
    seen = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            reasons.append(f"malformed_box:{line_number}")
            continue
        try:
            source_class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:5])
        except ValueError:
            reasons.append(f"malformed_box:{line_number}")
            continue
        decision_key = (
            (image_relative_path, line_number, source_class_id)
            if image_relative_path
            else None
        )
        review = (review_decisions or {}).get(decision_key) if decision_key else None
        if review and used_review_decisions is not None:
            used_review_decisions.add(decision_key)
        class_mapping = mapping.get(source_class_id)
        if review and dataset_source_id != review["source_id"]:
            raise ValueError(f"Review decision source does not match mapping for {image_relative_path}.")
        if review and review["decision"] == "drop":
            reasons.append(f"reviewed_drop:{line_number}")
            continue
        if not class_mapping and not review:
            unresolved = True
            reasons.append(f"unmapped_class:{source_class_id}")
            continue
        if review:
            target_name = review["target_class"]
            target_id = class_names().index(target_name)
            reasons.append(f"reviewed_{review['decision']}:{line_number}")
        elif class_mapping.get("action") == "drop":
            unresolved = True
            reasons.append(f"unmapped_or_dropped_class:{source_class_id}")
            continue
        elif class_mapping.get("target_name") == "accident":
            unresolved = True
            reasons.append(f"accident_requires_visual_review:{line_number}")
            continue
        else:
            target_id = class_mapping.get("target_id")
        if not isinstance(target_id, int) or not 0 <= target_id < len(class_names()):
            unresolved = True
            reasons.append(f"invalid_target_class:{source_class_id}")
            continue
        coords = (xc, yc, width, height)
        if not all(math.isfinite(value) for value in coords) or width <= 0 or height <= 0:
            reasons.append(f"invalid_box_geometry:{line_number}")
            continue
        if min(xc - width / 2, yc - height / 2) < 0 or max(xc + width / 2, yc + height / 2) > 1:
            reasons.append(f"out_of_range_box:{line_number}")
            continue
        if width * image_width < MIN_BOX_SIDE_PIXELS or height * image_height < MIN_BOX_SIDE_PIXELS:
            reasons.append(f"tiny_box:{line_number}")
            continue
        signature = (target_id, *(round(value, 6) for value in coords))
        if signature in seen:
            reasons.append(f"duplicate_box:{line_number}")
            continue
        seen.add(signature)
        boxes.append((target_id, xc, yc, width, height))
    return boxes, reasons, unresolved


def discover_records(
    raw_root: Path,
    provenance: dict[str, dict[str, str]],
    mappings: dict[str, dict[int, dict]],
    approved_sources: set[str],
    review_decisions: dict[tuple[str, int, int], dict] | None = None,
) -> list[ImageRecord]:
    image_paths = sorted(path for path in raw_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No source images found under {raw_root.relative_to(ROOT)}.")
    actual = {path.relative_to(raw_root).as_posix() for path in image_paths}
    missing = actual - provenance.keys()
    stale = provenance.keys() - actual
    if missing or stale:
        raise ValueError(
            f"Provenance coverage mismatch: {len(missing)} images lack rows; "
            f"{len(stale)} rows reference missing images."
        )

    records = []
    used_review_decisions: set[tuple[str, int, int]] = set()
    for path in image_paths:
        relative = path.relative_to(raw_root).as_posix()
        source = provenance[relative]
        if source["source_id"] not in approved_sources:
            raise ValueError(f"Source {source['source_id']!r} is not license-approved for training.")
        if source["source_id"] not in mappings:
            raise ValueError(f"No class mapping configured for source {source['source_id']!r}.")
        with Image.open(path) as image:
            width, height = image.size
        label = _label_path(path)
        boxes, reasons, unresolved = clean_yolo_label(
            label,
            mappings[source["source_id"]],
            width,
            height,
            relative,
            review_decisions,
            used_review_decisions,
            dataset_source_id=source["source_id"],
        )
        if unresolved:
            reasons.append("quarantined_for_label_review")
        records.append(
            ImageRecord(
                path=path,
                relative_path=relative,
                label_path=label,
                source_id=source["source_id"],
                group_id=source["group_id"],
                scene_id=source["scene_id"],
                boxes=boxes if not unresolved else [],
                phash=perceptual_hash(path),
                review_reasons=reasons,
            )
        )
    unused = set(review_decisions or {}) - used_review_decisions
    if unused:
        first = sorted(unused)[0]
        raise ValueError(
            f"Review decision does not match a source label: {first[0]} line {first[1]}."
        )
    return records


def load_embeddings(model_path: Path, records: list[ImageRecord]) -> None:
    """Compute DINOv2/CLIP embeddings using a caller-supplied local TorchScript encoder."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Local CLIP/DINOv2 TorchScript encoder not found: {model_path}")
    import torch

    encoder = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.inference_mode():
        for record in records:
            image = Image.open(record.path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            result = encoder(tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            vector = result.reshape(-1).cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm == 0 or not np.isfinite(norm):
                raise ValueError(f"Embedding encoder returned invalid vector for {record.relative_path}")
            record.embedding = vector / norm


def cluster_duplicates(
    records: list[ImageRecord],
    embedding_similarity: float = 0.96,
    phash_distance: int = 6,
    batch_size: int = 128,
) -> tuple[UnionFind, list[tuple[int, int, str]]]:
    if any(record.embedding is None for record in records):
        raise ValueError("CLIP/DINOv2 embeddings are required for every image.")
    union = UnionFind(len(records))
    edges: list[tuple[int, int, str]] = []
    hash_bits = np.stack([record.phash.ravel() for record in records]).astype(np.uint8)
    vectors = np.stack([record.embedding for record in records])
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        similarities = vectors[start:end] @ vectors.T
        distances = np.count_nonzero(hash_bits[start:end, None, :] != hash_bits[None, :, :], axis=2)
        for local_index in range(end - start):
            current = start + local_index
            candidates = np.flatnonzero(
                (similarities[local_index] >= embedding_similarity)
                | (distances[local_index] <= phash_distance)
            )
            for other in candidates:
                if int(other) <= current:
                    continue
                reason = "phash" if distances[local_index, other] <= phash_distance else "embedding"
                union.union(current, int(other))
                edges.append((current, int(other), reason))
    return union, edges


def assign_group_splits(
    records: list[ImageRecord], union: UnionFind, seed: int = 20261024
) -> None:
    group_roots: dict[str, int] = {}
    for index, record in enumerate(records):
        previous = group_roots.setdefault(record.group_id, index)
        union.union(previous, index)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        duplicate_root = union.find(index)
        grouped[f"{record.group_id}|duplicate_cluster:{duplicate_root}"].append(index)

    class_count = len(class_names())
    group_counts = {}
    for group_id, indices in grouped.items():
        counts = np.zeros(class_count, dtype=float)
        for index in indices:
            for box in records[index].boxes:
                counts[box[0]] += 1
        group_counts[group_id] = counts

    totals = sum(group_counts.values(), np.zeros(class_count, dtype=float))
    assigned = {split: np.zeros(class_count, dtype=float) for split in SPLIT_RATIOS}
    target = {split: totals * ratio for split, ratio in SPLIT_RATIOS.items()}
    split_group_counts = Counter()
    order = sorted(
        group_counts,
        key=lambda group_id: (-float(group_counts[group_id].max()), hashlib.sha256(group_id.encode()).hexdigest()),
    )
    rng = np.random.default_rng(seed)
    for group_id in order:
        counts = group_counts[group_id]
        candidates = []
        for split, ratio in SPLIT_RATIOS.items():
            proposed = assigned[split] + counts
            class_error = np.sum(((proposed - target[split]) / np.maximum(totals, 1)) ** 2)
            size_error = ((split_group_counts[split] + 1) / max(len(grouped), 1) - ratio) ** 2
            candidates.append((class_error + size_error, float(rng.random()), split))
        chosen = min(candidates)[2]
        assigned[chosen] += counts
        split_group_counts[chosen] += 1
        for index in grouped[group_id]:
            records[index].split = chosen


def overlap_violations(records: list[ImageRecord], union: UnionFind, edges: list[tuple[int, int, str]]) -> list[str]:
    violations = []
    hash_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[int, set[str]] = defaultdict(set)
    test_sources = set()
    for index, record in enumerate(records):
        digest = hashlib.sha256(record.phash.tobytes()).hexdigest()
        hash_splits[digest].add(record.split or "")
        cluster_splits[union.find(index)].add(record.split or "")
        if record.split == "test":
            test_sources.add(record.source_id)
    if any(len(splits) > 1 for splits in hash_splits.values()):
        violations.append("perceptual hash overlap across splits")
    if any(len(splits) > 1 for splits in cluster_splits.values()):
        violations.append("duplicate/near-duplicate cluster overlap across splits")
    near_duplicate_sources_in_train = set()
    for left, right, _ in edges:
        a, b = records[left], records[right]
        if a.split == "train" or b.split == "train":
            near_duplicate_sources_in_train.update((a.source_id, b.source_id))
    if test_sources & near_duplicate_sources_in_train:
        violations.append(
            "test includes a source that also contributed near-duplicates to train"
        )
    return violations


def split_counts(records: Iterable[ImageRecord]) -> dict[str, dict[str, dict[str, int]]]:
    records = list(records)
    result = {}
    for split in SPLIT_RATIOS:
        split_records = [record for record in records if record.split == split]
        per_class = {}
        for class_id, name in enumerate(class_names()):
            matched = [
                (record, box)
                for record in split_records
                for box in record.boxes
                if box[0] == class_id
            ]
            per_class[name] = {
                "images": len({record.relative_path for record, _ in matched}),
                "boxes": len(matched),
                "groups": len({record.group_id for record, _ in matched}),
                "scenes": len({record.scene_id for record, _ in matched if record.scene_id}),
            }
        result[split] = per_class
    return result


def write_stats(
    path: Path,
    counts: dict,
    violations: list[str],
    not_evaluable: list[str] | None = None,
) -> None:
    lines = [
        "# Rebuilt dataset statistics",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "All counts below are computed from the current build input. Classes with fewer than "
        f"{MIN_ACCIDENT_BOXES} ground-truth boxes in either validation or test are **NOT EVALUABLE**.",
        "",
        "| Split | Class | Images | Boxes | Groups | Scenes | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    accident_names = {item["name"] for item in load_classes() if item["group"] == "accident"}
    for split, classes in counts.items():
        for class_name, values in classes.items():
            evaluable = not (
                class_name in accident_names
                and split in {"val", "test"}
                and values["boxes"] < MIN_ACCIDENT_BOXES
            )
            status = "EVALUABLE" if evaluable else "NOT EVALUABLE: fewer than 30 boxes"
            lines.append(
                f"| {split} | {class_name} | {values['images']} | {values['boxes']} | "
                f"{values['groups']} | {values['scenes']} | {status} |"
            )
    lines.extend(["", "## Leakage validation", ""])
    lines.extend([f"- FAIL: {item}" for item in violations] or ["- PASS: no detected overlap violations"])
    lines.extend(["", "## Evaluable-class coverage", ""])
    lines.extend(
        [f"- NOT EVALUABLE: {item}" for item in not_evaluable or []]
        or ["- All accident classes meet the minimum validation and test box counts."]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet_candidates(
    records: list[ImageRecord],
    output_dir: Path,
    mappings: dict[str, dict[int, dict]],
) -> None:
    """Export up to 200 unresolved/suspicious images for each accident class."""
    from PIL import ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    accident_ids = [item["id"] for item in load_classes() if item["group"] == "accident"]
    decision_rows = []
    for record in records:
        if not record.review_reasons:
            continue
        source_mapping = mappings[record.source_id]
        if not record.label_path.is_file():
            continue
        for line_number, line in enumerate(record.label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            try:
                source_class_id = int(parts[0])
            except ValueError:
                continue
            mapping = source_mapping.get(source_class_id)
            if not mapping or mapping.get("target_name") == "accident" or mapping.get("action") == "drop":
                decision_rows.append(
                    {
                        "image": record.relative_path,
                        "source_id": record.source_id,
                        "source_class_id": source_class_id,
                        "line_number": line_number,
                    }
                )
    candidates_path = output_dir / "ambiguous_labels.csv"
    with candidates_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image", "source_id", "source_class_id", "line_number"),
        )
        writer.writeheader()
        writer.writerows(decision_rows)
    for class_id in accident_ids:
        candidates = [
            record for record in records
            if record.review_reasons and any("accident" in reason or "missing_label" in reason for reason in record.review_reasons)
        ][:200]
        if not candidates:
            continue
        thumb_size, columns = (240, 180), 5
        rows = (len(candidates) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 32)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(candidates):
            image = Image.open(record.path).convert("RGB")
            image.thumbnail(thumb_size)
            x, y = (index % columns) * thumb_size[0], (index // columns) * (thumb_size[1] + 32)
            sheet.paste(image, (x, y))
            draw.text((x + 2, y + thumb_size[1] + 2), record.relative_path[:38], fill="black")
        sheet.save(output_dir / f"accident_class_{class_id}_review.jpg", quality=90)


def write_test_manifest(output_root: Path, target: Path = TEST_MANIFEST_PATH) -> None:
    entries = []
    for path in sorted((output_root / "images" / "test").glob("*")):
        if path.is_file():
            label_path = output_root / "labels" / "test" / f"{path.stem}.txt"
            entries.append((sha256_file(path), path.relative_to(DATA_ROOT).as_posix()))
            entries.append((sha256_file(label_path), label_path.relative_to(DATA_ROOT).as_posix()))
    content = "# Frozen test split image and label SHA256 manifest\n" + "".join(
        f"{digest}  {relative}\n" for digest, relative in sorted(entries, key=lambda item: item[1])
    )
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("Frozen test manifest would change; do not overwrite the frozen test set.")
    target.write_text(content, encoding="utf-8")


def build_dataset(
    raw_root: Path = RAW_ROOT,
    provenance_path: Path = PROVENANCE_PATH,
    output_root: Path = OUTPUT_ROOT,
    embedding_model: Path | None = None,
    seed: int = 20261024,
) -> None:
    provenance = read_provenance(provenance_path)
    approved_sources = require_approved_sources()
    source_mappings = load_source_mappings()
    review_decisions = load_review_decisions()
    records = discover_records(
        raw_root,
        provenance,
        source_mappings,
        approved_sources,
        review_decisions,
    )
    if embedding_model is None:
        raise ValueError(
            "A local CLIP/DINOv2 TorchScript embedding model is required. "
            "Export the approved encoder locally; this script will not fetch model weights."
        )
    load_embeddings(embedding_model, records)
    union, edges = cluster_duplicates(records)
    assign_group_splits(records, union, seed=seed)
    violations = overlap_violations(records, union, edges)
    counts = split_counts(records)
    accident_names = {item["name"] for item in load_classes() if item["group"] == "accident"}
    not_evaluable = []
    for name in accident_names:
        for split in ("val", "test"):
            if counts[split][name]["boxes"] < MIN_ACCIDENT_BOXES:
                not_evaluable.append(
                    f"{name} has {counts[split][name]['boxes']} {split} ground-truth boxes "
                    f"(< {MIN_ACCIDENT_BOXES}); marked NOT EVALUABLE"
                )
    unresolved_images = [
        record.relative_path
        for record in records
        if "quarantined_for_label_review" in (record.review_reasons or [])
        or "missing_label_file" in (record.review_reasons or [])
    ]
    if unresolved_images:
        violations.append(
            f"{len(unresolved_images)} images require label review or missing-label resolution; "
            "review contact sheets before rebuilding"
        )

    write_stats(DATA_ROOT / "dataset_stats.md", counts, violations, not_evaluable)
    write_contact_sheet_candidates(
        records,
        DATA_ROOT / "label_review" / "contact_sheets",
        source_mappings,
    )
    if violations:
        raise ValueError("Dataset build failed hard requirements; see ml/data/dataset_stats.md.")

    with tempfile.TemporaryDirectory(prefix="acs_dataset_", dir=DATA_ROOT) as temporary:
        staging = Path(temporary) / "merged"
        for record in records:
            if not record.split:
                raise ValueError(f"No split assigned to {record.relative_path}")
            stem = hashlib.sha256(record.relative_path.encode()).hexdigest()[:20]
            image_dir = staging / "images" / record.split
            label_dir = staging / "labels" / record.split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.path, image_dir / f"{stem}{record.path.suffix.lower()}")
            label_file = label_dir / f"{stem}.txt"
            label_file.write_text(
                "".join(
                    f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}\n"
                    for class_id, xc, yc, width, height in record.boxes
                ),
                encoding="utf-8",
            )
        data_yaml = {"path": str(output_root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": class_names(), "nc": len(class_names())}
        (staging / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        if output_root.exists():
            raise FileExistsError(f"{output_root} already exists; archive it explicitly before replacing.")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(output_root))
    write_test_manifest(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE_PATH)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    build_dataset(
        raw_root=args.raw_dir,
        provenance_path=args.provenance,
        embedding_model=args.embedding_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
