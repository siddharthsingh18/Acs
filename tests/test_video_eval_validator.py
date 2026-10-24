import csv
from pathlib import Path

from ml.scripts.validate_video_eval import REQUIRED_FIELDS, validate_video_labels


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def negative_clip(clip_id: str, split: str, local_path: str) -> dict[str, str]:
    return {
        "clip_id": clip_id,
        "camera_id": f"camera-{clip_id}",
        "scene_id": f"scene-{clip_id}",
        "start_seconds": "none",
        "end_seconds": "none",
        "accident_type": "none",
        "conditions": "night,rain,glare",
        "license": "CC BY 4.0",
        "source_url": f"https://example.test/{clip_id}",
        "capture_start_utc": "2026-01-01T00:00:00Z",
        "duration_seconds": "3600",
        "contains_accident": "false",
        "split": split,
        "local_path": local_path,
    }


def test_video_label_validator_rejects_missing_required_fields(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    row = negative_clip("a", "test", "missing.mp4")
    row["license"] = ""
    write_labels(labels, [row])
    errors, _ = validate_video_labels(labels, tmp_path)
    assert any("empty required fields: license" in error for error in errors)


def test_video_label_validator_rejects_camera_or_scene_split_leakage(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    rows = [
        negative_clip("a", "test", "missing-a.mp4"),
        negative_clip("b", "tuning", "missing-b.mp4"),
    ]
    rows[1]["camera_id"] = rows[0]["camera_id"]
    write_labels(labels, rows)
    errors, _ = validate_video_labels(labels, tmp_path)
    assert any("occurs in both tuning and test" in error for error in errors)


def test_video_labels_count_unique_accident_free_camera_hours(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    first = negative_clip("a", "test", "missing-a.mp4")
    second = negative_clip("b", "test", "missing-b.mp4")
    second["camera_id"] = first["camera_id"]
    second["capture_start_utc"] = "2026-01-01T01:00:00Z"
    write_labels(labels, [first, second])
    errors, stats = validate_video_labels(labels, tmp_path)
    assert not any("overlapping clips" in error for error in errors)
    assert stats["accident_free_hours"] == 2.0
    assert stats["test_accident_free_hours"] == 2.0
    assert stats["cameras"] == 1
