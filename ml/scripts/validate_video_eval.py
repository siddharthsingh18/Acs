#!/usr/bin/env python3
"""Validate video-evaluation labels, camera/scene splits, and evidence targets."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "ml" / "data" / "video_eval" / "labels.csv"
REQUIRED_FIELDS = {
    "clip_id",
    "camera_id",
    "scene_id",
    "start_seconds",
    "end_seconds",
    "accident_type",
    "conditions",
    "license",
    "source_url",
    "capture_start_utc",
    "duration_seconds",
    "contains_accident",
    "split",
    "local_path",
}
MIN_EVENTS = 40
MIN_CAMERAS_OR_SCENES = 15
MIN_ACCIDENT_FREE_HOURS = 20.0
MIN_TEST_EVENTS = 40
MIN_TEST_CAMERAS_OR_SCENES = 15
MIN_TEST_ACCIDENT_FREE_HOURS = 20.0


def validate_video_labels(
    csv_path: Path = LABELS,
    video_root: Path = LABELS.parent,
) -> tuple[list[str], dict[str, float | int]]:
    errors = []
    if not csv_path.is_file():
        raise FileNotFoundError(f"Video evaluation labels are missing: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing_headers = REQUIRED_FIELDS - fields
        if missing_headers:
            raise ValueError(f"labels.csv is missing columns: {', '.join(sorted(missing_headers))}")
        rows = list(reader)

    clip_ids = set()
    split_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    event_count = 0
    accident_cameras = set()
    accident_scenes = set()
    test_events = 0
    test_accident_cameras = set()
    test_accident_scenes = set()
    accident_free_seconds = 0.0
    test_accident_free_seconds = 0.0
    intervals: dict[str, list[tuple[float, float, bool, int]]] = defaultdict(list)
    for line_number, row in enumerate(rows, 2):
        missing_values = sorted(name for name in REQUIRED_FIELDS if not (row.get(name) or "").strip())
        if missing_values:
            errors.append(f"row {line_number}: empty required fields: {', '.join(missing_values)}")
            continue
        clip_id = row["clip_id"].strip()
        if clip_id in clip_ids:
            errors.append(f"row {line_number}: duplicate clip_id {clip_id}")
        clip_ids.add(clip_id)
        if row["split"] not in {"tuning", "test"}:
            errors.append(f"row {line_number}: split must be tuning or test")
        for dimension in ("camera_id", "scene_id"):
            split_groups[(dimension, row[dimension])].add(row["split"])
        try:
            duration = float(row["duration_seconds"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"row {line_number}: duration_seconds must be positive and finite")
            continue
        local_path = (video_root / row["local_path"]).resolve()
        if not local_path.is_relative_to(video_root.resolve()) or not local_path.is_file():
            errors.append(f"row {line_number}: clip file missing or outside video_eval: {row['local_path']}")
        url = urlparse(row["source_url"])
        if url.scheme not in {"http", "https"} or not url.netloc:
            errors.append(f"row {line_number}: source_url must be a public HTTP(S) URL")
        try:
            capture_start = datetime.fromisoformat(row["capture_start_utc"].replace("Z", "+00:00"))
            if capture_start.tzinfo is None:
                raise ValueError
            capture_epoch = capture_start.astimezone(timezone.utc).timestamp()
        except ValueError:
            errors.append(f"row {line_number}: capture_start_utc must be ISO-8601 with a timezone")
            continue
        if row["contains_accident"].lower() not in {"true", "false"}:
            errors.append(f"row {line_number}: contains_accident must be true or false")
            continue
        intervals[row["camera_id"]].append(
            (capture_epoch, capture_epoch + duration, row["contains_accident"].lower() == "true", line_number)
        )
        if row["contains_accident"].lower() == "true":
            event_count += 1
            accident_cameras.add(row["camera_id"])
            accident_scenes.add(row["scene_id"])
            if row["split"] == "test":
                test_events += 1
                test_accident_cameras.add(row["camera_id"])
                test_accident_scenes.add(row["scene_id"])
            try:
                start, end = float(row["start_seconds"]), float(row["end_seconds"])
                if not (0 <= start < end <= duration):
                    raise ValueError
            except ValueError:
                errors.append(f"row {line_number}: accident interval must satisfy 0 <= start < end <= duration")
        else:
            if row["start_seconds"].lower() != "none" or row["end_seconds"].lower() != "none":
                errors.append(f"row {line_number}: accident-free clips must use 'none' for event interval")
            if row["accident_type"].lower() != "none":
                errors.append(f"row {line_number}: accident-free clips must use accident_type=none")
            accident_free_seconds += duration
            if row["split"] == "test":
                test_accident_free_seconds += duration

    camera_free_hours = 0.0
    for camera_id, camera_intervals in intervals.items():
        ordered = sorted(camera_intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                errors.append(
                    f"camera {camera_id!r}: overlapping clips on rows {previous[3]} and {current[3]}; "
                    "overlapping footage cannot be counted twice"
                )
        camera_free_hours += sum(
            end - start
            for start, end, contains_accident, _ in camera_intervals
            if not contains_accident
        ) / 3600.0

    for (dimension, identity), splits in split_groups.items():
        if len(splits) > 1:
            errors.append(f"{dimension} {identity!r} occurs in both tuning and test")
    group_count = len({row["scene_id"] for row in rows})
    test_group_count = len({row["scene_id"] for row in rows if row["split"] == "test"})
    test_ratio = test_group_count / group_count if group_count else 0.0
    if rows and abs(test_ratio - 0.70) > 0.05:
        errors.append(f"test scene share is {test_ratio:.1%}; expected 70% ±5%")

    stats: dict[str, float | int] = {
        "clips": len(rows),
        "events": event_count,
        "accident_cameras": len(accident_cameras),
        "accident_scenes": len(accident_scenes),
        "test_events": test_events,
        "test_accident_cameras": len(test_accident_cameras),
        "test_accident_scenes": len(test_accident_scenes),
        "accident_free_hours": camera_free_hours,
        "test_accident_free_hours": test_accident_free_seconds / 3600.0,
        "accident_free_clip_hours": accident_free_seconds / 3600.0,
        "cameras": len(intervals),
        "scenes": group_count,
        "test_scene_share": test_ratio,
    }
    return errors, stats


def write_shortfall_report(stats: dict[str, float | int], errors: list[str], output_path: Path) -> None:
    events = int(stats["events"])
    cameras = int(stats["accident_cameras"])
    scenes = int(stats["accident_scenes"])
    hours = float(stats["accident_free_hours"])
    lines = [
        "# Video evaluation evidence and shortfall",
        "",
        "Status: **PRELIMINARY — no production-grade frozen evaluation set is established.**",
        "",
        "| Target | Required | Available | Shortfall |",
        "|---|---:|---:|---:|",
        f"| Distinct accident events | {MIN_EVENTS} | {events} | {max(0, MIN_EVENTS - events)} |",
        f"| Distinct accident cameras or scenes | {MIN_CAMERAS_OR_SCENES} | {max(cameras, scenes)} | {max(0, MIN_CAMERAS_OR_SCENES - max(cameras, scenes))} |",
        f"| Accident-free hours | {MIN_ACCIDENT_FREE_HOURS:.1f} | {hours:.3f} | {max(0.0, MIN_ACCIDENT_FREE_HOURS - hours):.3f} |",
        f"| Clips with complete labels and local files | 1+ | {stats['clips']} | {max(0, 1 - int(stats['clips']))} |",
        "",
        "## Frozen test-set promotion minimums",
        "",
        "| Test-only target | Required | Available | Shortfall |",
        "|---|---:|---:|---:|",
        f"| Distinct accident events | {MIN_TEST_EVENTS} | {stats['test_events']} | {max(0, MIN_TEST_EVENTS - int(stats['test_events']))} |",
        f"| Accident cameras or scenes | {MIN_TEST_CAMERAS_OR_SCENES} | {max(int(stats['test_accident_cameras']), int(stats['test_accident_scenes']))} | {max(0, MIN_TEST_CAMERAS_OR_SCENES - max(int(stats['test_accident_cameras']), int(stats['test_accident_scenes'])))} |",
        f"| Accident-free camera-hours | {MIN_TEST_ACCIDENT_FREE_HOURS:.1f} | {float(stats['test_accident_free_hours']):.3f} | {max(0.0, MIN_TEST_ACCIDENT_FREE_HOURS - float(stats['test_accident_free_hours'])):.3f} |",
        "",
        "The frozen test target is 70% of camera/scene groups; tuning receives 30%. No clip from a camera or scene may cross splits.",
        "Only footage with verified permission for local storage and evaluation is admissible. Do not use access-control bypasses.",
        "",
        "## Validation errors",
        "",
        *([f"- {error}" for error in errors] or ["- None."]),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--video-root", type=Path, default=LABELS.parent)
    parser.add_argument("--report", type=Path, default=ROOT / "ml" / "data" / "VIDEO_EVAL_STATUS.md")
    args = parser.parse_args()
    errors, stats = validate_video_labels(args.labels, args.video_root)
    write_shortfall_report(stats, errors, args.report)
    if (
        errors
        or int(stats["events"]) < MIN_EVENTS
        or max(int(stats["accident_cameras"]), int(stats["accident_scenes"])) < MIN_CAMERAS_OR_SCENES
        or float(stats["accident_free_hours"]) < MIN_ACCIDENT_FREE_HOURS
        or int(stats["test_events"]) < MIN_TEST_EVENTS
        or max(int(stats["test_accident_cameras"]), int(stats["test_accident_scenes"]))
        < MIN_TEST_CAMERAS_OR_SCENES
        or float(stats["test_accident_free_hours"]) < MIN_TEST_ACCIDENT_FREE_HOURS
    ):
        raise SystemExit(f"Video evaluation evidence is below target; see {args.report}.")


if __name__ == "__main__":
    main()
