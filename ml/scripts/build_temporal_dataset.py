#!/usr/bin/env python3
"""
ml/scripts/build_temporal_dataset.py
Extracts sliding-window track pair features from labeled video clips to create ml/data/temporal/.

Features extracted:
1. Bounding-box overlap (IoU) trend (max, mean, slope, final, delta)
2. Closing speed (min distance, max closing speed, mean closing speed, final distance)
3. Deceleration (max deceleration, mean deceleration, post-approach deceleration, speed drop)
4. Direction change angle (max angular deflection, heading convergence, heading deflection)
5. Time-to-collision (min TTC, mean TTC, critical TTC fraction)
6. Post-overlap stationary time (stationary duration, ratio, stopped at end)
7. Number of vehicles stopped (in track pair and across the scene)

Data split policy:
- Strictly split by video, NEVER by window.
- Tuning videos (pos + neg) -> Train/Val split.
- Test holdout videos (test_holdout_acc_ex.mp4, test_holdout_neg_traffic.mp4) -> Test split.

Preliminary Data Disclosure:
- If fewer than 100 labeled accident windows exist, logs and metadata report results as preliminary.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.detector import AccidentDetector
from acs.temporal.features import (
    FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    TrackHistoryBuffer,
    TrackPairFeatureExtractor,
    TrackPoint,
    TrackTrajectory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_temporal_dataset")

# Ground truth accident temporal start frame per positive clip
POSITIVE_CLIPS_START_FRAMES: Dict[str, int] = {
    "test_holdout_acc_ex.mp4": 100,
    "tuning_pos_cadp_crash.mp4": 15,
    "tuning_pos_cctv_crash.mp4": 20,
}


def generate_synthetic_hard_negatives(
    count_per_type: int = 15,
    fps: float = 30.0,
    window_seconds: float = 1.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Synthesizes physically realistic hard-negative trajectory pairs:
    1. U-turns: Large direction change angle (~180 deg) with smooth continuous movement (no stop).
    2. Close passes: Two vehicles passing each other closely at high speed, zero contact, zero stop.
    3. Sharp braking: Vehicle decelerating sharply to avoid obstacle or stop at light, but without crash.
    4. Congested creeping: Vehicles moving bumper-to-bumper at low speed without impact.
    """
    rng = np.random.RandomState(seed)
    extractor = TrackPairFeatureExtractor()
    samples: List[Dict[str, Any]] = []

    num_frames = int(window_seconds * fps)

    # 1. U-Turns
    for idx in range(count_per_type):
        t_a = TrackTrajectory(1000 + idx * 2)
        t_b = TrackTrajectory(1001 + idx * 2)

        start_x = rng.uniform(0.3, 0.7)
        start_y = rng.uniform(0.2, 0.4)
        radius = rng.uniform(0.08, 0.15)
        speed_norm = rng.uniform(0.05, 0.12)  # moving smoothly

        # Other vehicle traveling straight in parallel lane
        other_x = start_x + rng.uniform(-0.25, -0.15)

        for f in range(num_frames):
            t = f / fps
            angle = math.pi * (f / num_frames)  # 180 degree U-turn arc
            xa = start_x + radius * math.sin(angle)
            ya = start_y + radius * (1.0 - math.cos(angle))

            xb = other_x
            yb = start_y + speed_norm * t

            t_a.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(xa * 640, ya * 480, (xa + 0.08) * 640, (ya + 0.08) * 480),
                norm_bbox=(xa, ya, xa + 0.08, ya + 0.08),
                centroid=(xa + 0.04, ya + 0.04),
                confidence=0.88, class_id=2, class_name="car", track_id=1000 + idx * 2,
            ))
            t_b.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(xb * 640, yb * 480, (xb + 0.08) * 640, (yb + 0.08) * 480),
                norm_bbox=(xb, yb, xb + 0.08, yb + 0.08),
                centroid=(xb + 0.04, yb + 0.04),
                confidence=0.88, class_id=2, class_name="car", track_id=1001 + idx * 2,
            ))

        res = extractor.extract_pair_features(t_a, t_b, window_seconds=window_seconds, fps=fps)
        if res is not None:
            feats, tab_v, seq_m = res
            samples.append({
                "video_name": "synthetic_uturn",
                "split": "train",
                "label": 0,
                "category": "hard_negative_uturn",
                "features": feats,
                "tabular_vector": tab_v,
                "sequence_matrix": seq_m,
            })

    # 2. Close Passes
    for idx in range(count_per_type):
        t_a = TrackTrajectory(2000 + idx * 2)
        t_b = TrackTrajectory(2001 + idx * 2)

        lane_gap = rng.uniform(0.06, 0.11)  # close lateral distance
        speed_a = rng.uniform(0.15, 0.28)
        speed_b = rng.uniform(0.12, 0.22)

        for f in range(num_frames):
            t = f / fps
            xa = 0.45
            ya = 0.1 + speed_a * t

            xb = 0.45 + lane_gap
            yb = 0.9 - speed_b * t  # oncoming or overtaking

            t_a.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(xa * 640, ya * 480, (xa + 0.06) * 640, (ya + 0.06) * 480),
                norm_bbox=(xa, ya, xa + 0.06, ya + 0.06),
                centroid=(xa + 0.03, ya + 0.03),
                confidence=0.90, class_id=2, class_name="car", track_id=2000 + idx * 2,
            ))
            t_b.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(xb * 640, yb * 480, (xb + 0.06) * 640, (yb + 0.06) * 480),
                norm_bbox=(xb, yb, xb + 0.06, yb + 0.06),
                centroid=(xb + 0.03, yb + 0.03),
                confidence=0.90, class_id=2, class_name="car", track_id=2001 + idx * 2,
            ))

        res = extractor.extract_pair_features(t_a, t_b, window_seconds=window_seconds, fps=fps)
        if res is not None:
            feats, tab_v, seq_m = res
            samples.append({
                "video_name": "synthetic_close_pass",
                "split": "train",
                "label": 0,
                "category": "hard_negative_close_pass",
                "features": feats,
                "tabular_vector": tab_v,
                "sequence_matrix": seq_m,
            })

    # 3. Braking to stop (single track stopping behind traffic, no impact)
    for idx in range(count_per_type):
        t_a = TrackTrajectory(3000 + idx * 2)
        t_b = TrackTrajectory(3001 + idx * 2)

        # Vehicle B is stopped ahead at red light
        xb = 0.50
        yb = 0.65

        # Vehicle A approaches, brakes hard, stops 0.12 units behind B (no collision)
        init_speed = rng.uniform(0.18, 0.30)
        stop_time = window_seconds * rng.uniform(0.5, 0.75)

        for f in range(num_frames):
            t = f / fps
            xb_curr = xb
            yb_curr = yb

            if t < stop_time:
                decel = init_speed / stop_time
                curr_speed = max(0.0, init_speed - decel * t)
                ya = 0.2 + init_speed * t - 0.5 * decel * (t ** 2)
            else:
                curr_speed = 0.0
                ya = 0.2 + 0.5 * init_speed * stop_time

            # Ensure safe buffer
            ya = min(ya, yb - 0.11)

            t_a.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(0.5 * 640, ya * 480, 0.56 * 640, (ya + 0.06) * 480),
                norm_bbox=(0.5, ya, 0.56, ya + 0.06),
                centroid=(0.53, ya + 0.03),
                confidence=0.92, class_id=2, class_name="car", track_id=3000 + idx * 2,
            ))
            t_b.add_point(TrackPoint(
                frame_idx=f, timestamp=t,
                bbox=(0.5 * 640, yb_curr * 480, 0.56 * 640, (yb_curr + 0.06) * 480),
                norm_bbox=(0.5, yb_curr, 0.56, yb_curr + 0.06),
                centroid=(0.53, yb_curr + 0.03),
                confidence=0.92, class_id=2, class_name="car", track_id=3001 + idx * 2,
            ))

        res = extractor.extract_pair_features(t_a, t_b, window_seconds=window_seconds, fps=fps)
        if res is not None:
            feats, tab_v, seq_m = res
            samples.append({
                "video_name": "synthetic_braking",
                "split": "train",
                "label": 0,
                "category": "hard_negative_braking",
                "features": feats,
                "tabular_vector": tab_v,
                "sequence_matrix": seq_m,
            })

    return samples


def extract_features_from_video(
    video_path: Path,
    split: str,
    detector: AccidentDetector,
    window_seconds_list: List[float] = [1.0, 1.5, 2.0],
    step_frames: int = 2,
) -> List[Dict[str, Any]]:
    """Runs ByteTrack tracker over a video clip and extracts sliding-window track pair features."""
    v_name = video_path.name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crash_start_frame = POSITIVE_CLIPS_START_FRAMES.get(v_name, -1)
    is_positive_video = crash_start_frame >= 0

    logger.info(
        "Processing %s [%s] (%d frames, %.1f fps, crash_start=%s)",
        v_name, split, total_frames, fps,
        str(crash_start_frame) if is_positive_video else "None",
    )

    history_buffer = TrackHistoryBuffer(max_buffer_seconds=4.0, fps=fps)
    all_window_samples: List[Dict[str, Any]] = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Track frame
        det_result = detector.track_frame(frame)
        ann_h, ann_w = det_result.annotated_frame.shape[:2]
        timestamp = frame_idx / fps

        history_buffer.add_frame_detections(
            frame_idx=frame_idx,
            timestamp=timestamp,
            detections=det_result.detections,
            frame_width=ann_w,
            frame_height=ann_h,
        )

        # Sample sliding windows at step_frames interval
        if frame_idx % step_frames == 0:
            for win_sec in window_seconds_list:
                win_frames = int(win_sec * fps)
                if frame_idx < win_frames:
                    continue

                # Ground truth window label:
                # If window extends into crash phase (at least 5 frames beyond crash onset)
                is_accident_window = is_positive_video and (frame_idx >= crash_start_frame + 5)

                # Extract candidate pair interaction
                pair_result = history_buffer.extract_candidate_pair_features(
                    frame_width=ann_w,
                    frame_height=ann_h,
                    window_seconds=win_sec,
                )

                if pair_result is not None:
                    feats, tab_v, seq_m, (tid_a, tid_b) = pair_result
                    category = "accident" if is_accident_window else "normal_traffic"
                    if not is_positive_video:
                        if "traffic_jam" in v_name:
                            category = "hard_negative_traffic_jam"
                        elif "rain" in v_name:
                            category = "hard_negative_rain"
                        elif "night" in v_name:
                            category = "hard_negative_night"
                        else:
                            category = "normal_driving"

                    sample = {
                        "video_name": v_name,
                        "split": split,
                        "label": 1 if is_accident_window else 0,
                        "category": category,
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "window_seconds": win_sec,
                        "track_pair": [tid_a, tid_b],
                        "features": feats,
                        "tabular_vector": tab_v,
                        "sequence_matrix": seq_m,
                    }
                    all_window_samples.append(sample)

        frame_idx += 1

    cap.release()
    logger.info("  Extracted %d window samples from %s", len(all_window_samples), v_name)
    return all_window_samples


def build_dataset(
    video_root: Path,
    output_dir: Path,
    model_path: str = "best.pt",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Builds ml/data/temporal/ dataset from video evaluation clips."""
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = AccidentDetector(model_path=model_path, device=device)

    # 1. Tuning videos -> Train / Val split
    tuning_dir = video_root / "tuning"
    tuning_clips = sorted(tuning_dir.glob("*.mp4"))

    # 2. Test videos -> Test holdout split (NEVER mixed with train)
    test_dir = video_root / "test"
    test_clips = sorted(test_dir.glob("*.mp4"))

    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    logger.info("Building Train Split from %d tuning video clips...", len(tuning_clips))
    for v_path in tuning_clips:
        samples = extract_features_from_video(v_path, split="train", detector=detector)
        train_samples.extend(samples)

    # Add synthetic hard negatives (U-turns, close passes, sudden braking) to train
    logger.info("Generating synthetic hard negatives (U-turns, close passes, braking)...")
    synthetic_samples = generate_synthetic_hard_negatives(count_per_type=20, fps=30.0)
    train_samples.extend(synthetic_samples)
    logger.info("  Added %d synthetic hard-negative samples to train split.", len(synthetic_samples))

    logger.info("Building Test Split from %d holdout test video clips...", len(test_clips))
    for v_path in test_clips:
        samples = extract_features_from_video(v_path, split="test", detector=detector)
        test_samples.extend(samples)

    # Count labeled accident windows
    total_acc_train = sum(1 for s in train_samples if s["label"] == 1)
    total_acc_test = sum(1 for s in test_samples if s["label"] == 1)
    total_acc_windows = total_acc_train + total_acc_test

    total_neg_train = len(train_samples) - total_acc_train
    total_neg_test = len(test_samples) - total_acc_test

    is_preliminary = total_acc_windows < 100
    preliminary_notice = (
        f"PRELIMINARY RESULTS NOTICE: There are {total_acc_windows} labeled accident windows "
        f"across all clips (fewer than 100). Per evaluation criteria, all results and performance "
        f"estimates are formally designated as PRELIMINARY."
        if is_preliminary
        else "Dataset contains >= 100 labeled accident windows."
    )

    if is_preliminary:
        logger.warning(preliminary_notice)
    else:
        logger.info(preliminary_notice)

    # Pack into numpy arrays
    X_train_tab = np.array([s["tabular_vector"] for s in train_samples], dtype=np.float32)
    X_train_seq = np.array([s["sequence_matrix"] for s in train_samples], dtype=np.float32)
    y_train = np.array([s["label"] for s in train_samples], dtype=np.int64)

    X_test_tab = np.array([s["tabular_vector"] for s in test_samples], dtype=np.float32)
    X_test_seq = np.array([s["sequence_matrix"] for s in test_samples], dtype=np.float32)
    y_test = np.array([s["label"] for s in test_samples], dtype=np.int64)

    # Save to disk
    train_npz_path = output_dir / "train.npz"
    test_npz_path = output_dir / "test.npz"

    np.savez_compressed(
        train_npz_path,
        X_tab=X_train_tab,
        X_seq=X_train_seq,
        y=y_train,
        videos=np.array([s["video_name"] for s in train_samples]),
        categories=np.array([s["category"] for s in train_samples]),
    )

    np.savez_compressed(
        test_npz_path,
        X_tab=X_test_tab,
        X_seq=X_test_seq,
        y=y_test,
        videos=np.array([s["video_name"] for s in test_samples]),
        categories=np.array([s["category"] for s in test_samples]),
    )

    summary = {
        "dataset_dir": str(output_dir),
        "split_policy": "Split strictly by video, never by window",
        "preliminary_status": is_preliminary,
        "preliminary_notice": preliminary_notice,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "sequence_features": SEQUENCE_FEATURE_NAMES,
        "train": {
            "total_windows": len(train_samples),
            "accident_windows": total_acc_train,
            "normal_windows": total_neg_train,
            "videos": list(sorted(set(s["video_name"] for s in train_samples))),
        },
        "test": {
            "total_windows": len(test_samples),
            "accident_windows": total_acc_test,
            "normal_windows": total_neg_test,
            "videos": list(sorted(set(s["video_name"] for s in test_samples))),
        },
        "total_accident_windows": total_acc_windows,
        "total_windows": len(train_samples) + len(test_samples),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset successfully built and saved to %s", output_dir)
    logger.info(
        "Train split: %d windows (%d acc, %d neg) | Test split: %d windows (%d acc, %d neg)",
        len(train_samples), total_acc_train, total_neg_train,
        len(test_samples), total_acc_test, total_neg_test,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal tracking accident dataset.")
    parser.add_argument("--video-root", type=str, default="ml/data/video_eval", help="Root dir of video_eval clips")
    parser.add_argument("--output-dir", type=str, default="ml/data/temporal", help="Output directory")
    parser.add_argument("--model-path", type=str, default="best.pt", help="YOLO model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)")
    args = parser.parse_args()

    build_dataset(
        video_root=PROJECT_ROOT / args.video_root,
        output_dir=PROJECT_ROOT / args.output_dir,
        model_path=args.model_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
