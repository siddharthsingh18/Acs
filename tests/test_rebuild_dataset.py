from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.scripts.rebuild_dataset import (
    ImageRecord,
    UnionFind,
    assign_group_splits,
    clean_yolo_label,
    overlap_violations,
    perceptual_hash,
    read_provenance,
    require_approved_sources,
)


def test_provenance_requires_group_and_unique_path(tmp_path: Path):
    provenance = tmp_path / "provenance.csv"
    provenance.write_text(
        "relative_path,source_id,video_id,camera_id,scene_id\n"
        "a.jpg,source-1,,,scene-1\n"
        "b.jpg,source-1,,camera-1,\n",
        encoding="utf-8",
    )
    rows = read_provenance(provenance)
    assert rows["a.jpg"]["group_id"] == "group:scene-1"
    assert rows["b.jpg"]["group_id"] == "group:camera-1"

    provenance.write_text(
        "relative_path,source_id,video_id,camera_id,scene_id\na.jpg,source-1,,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No video_id"):
        read_provenance(provenance)


def test_provenance_rejects_duplicate_paths(tmp_path: Path):
    provenance = tmp_path / "provenance.csv"
    provenance.write_text(
        "relative_path,source_id,video_id,camera_id,scene_id\n"
        "a.jpg,s1,v1,,\na.jpg,s1,v1,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate provenance path"):
        read_provenance(provenance)


def test_clean_labels_drops_bad_tiny_and_duplicate_boxes(tmp_path: Path):
    labels = tmp_path / "image.txt"
    labels.write_text(
        "0 0.5 0.5 0.5 0.5\n"
        "0 0.5 0.5 0.5 0.5\n"
        "0 1.2 0.5 0.2 0.2\n"
        "0 0.5 0.5 0.001 0.001\n"
        "0 nan 0.5 0.1 0.1\n",
        encoding="utf-8",
    )
    mapping = {0: {"action": "map", "target_id": 1, "target_name": "car"}}
    boxes, reasons, unresolved = clean_yolo_label(labels, mapping, 1000, 1000)
    assert len(boxes) == 1
    assert "duplicate_box:2" in reasons
    assert "out_of_range_box:3" in reasons
    assert "tiny_box:4" in reasons
    assert "invalid_box_geometry:5" in reasons
    assert not unresolved


def test_generic_accident_is_quarantined_until_review(tmp_path: Path):
    labels = tmp_path / "image.txt"
    labels.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    mapping = {0: {"action": "map", "target_id": 3, "target_name": "accident"}}
    boxes, reasons, unresolved = clean_yolo_label(labels, mapping, 1000, 1000)
    assert boxes == []
    assert unresolved
    assert "accident_requires_visual_review:1" in reasons


def test_human_review_decision_is_explicitly_applied(tmp_path: Path):
    labels = tmp_path / "image.txt"
    labels.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    decision_key = ("scene/image.jpg", 1, 7)
    decisions = {
        decision_key: {
            "decision": "fix",
            "target_class": "accident",
            "reason": "Visible collision at the labeled location.",
            "source_id": "source-1",
        }
    }
    used = set()
    boxes, reasons, unresolved = clean_yolo_label(
        labels,
        {},
        1000,
        1000,
        "scene/image.jpg",
        decisions,
        used,
        "source-1",
    )
    assert boxes == [(3, 0.5, 0.5, 0.2, 0.2)]
    assert reasons == ["reviewed_fix:1"]
    assert not unresolved
    assert used == {decision_key}


def test_group_split_keeps_same_camera_and_near_duplicates_together(tmp_path: Path):
    records = []
    for index, (group, source) in enumerate(
        [("cam-a", "src-a"), ("cam-a", "src-a"), ("cam-b", "src-b"), ("cam-c", "src-c")]
    ):
        image = tmp_path / f"{index}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image)
        records.append(
            ImageRecord(
                path=image,
                relative_path=image.name,
                label_path=image.with_suffix(".txt"),
                source_id=source,
                group_id=group,
                scene_id=group,
                boxes=[(3, 0.5, 0.5, 0.1, 0.1)],
                phash=perceptual_hash(image),
                embedding=np.array([1.0, float(index)], dtype=np.float32),
            )
        )
    union = UnionFind(len(records))
    union.union(0, 1)
    assign_group_splits(records, union, seed=3)
    assert records[0].split == records[1].split
    assert not overlap_violations(records, union, [(0, 1, "embedding")])


def test_current_sources_are_not_enabled_without_verified_rights():
    with pytest.raises(ValueError, match="No sources are licensed and approved"):
        require_approved_sources()
