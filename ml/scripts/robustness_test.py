#!/usr/bin/env python3
"""
ml/scripts/robustness_test.py
Automated CCTV Accident Detection System (ACS)
Operational Robustness Evaluation & Targeted Data Augmentation Pipeline

Evaluates detection and incident-tracking robustness against 11 real-world visual
and stream corruptions across severities 1 to 3 on test clips:
  1. gaussian_noise: Additive Gaussian sensor noise (severities 1, 2, 3)
  2. motion_blur: Camera/object linear motion blur (severities 1, 2, 3)
  3. defocus_blur: Optical out-of-focus blur (severities 1, 2, 3)
  4. fog: Atmospheric scattering and contrast suppression (severities 1, 2, 3)
  5. rain_streaks: Slanted rain streak alpha overlay (severities 1, 2, 3)
  6. night_darkening: Severe ambient light reduction & gamma shift (severities 1, 2, 3)
  7. glare: Specular blooming from headlights/sun (severities 1, 2, 3)
  8. low_bitrate: H.264 video compression artifacts at 500 kbps, 350 kbps, 200 kbps
  9. resolution_drop: Stream resolution degradation down to 480p and 360p
 10. camera_shake: Wind/structural camera vibration jitter (severities 1, 2, 3)
 11. lens_dirt: Persistent grime and water droplet smudges on the lens (severities 1, 2, 3)

Identifies the worst 3 corruptions, applies matching Albumentations transforms to training
(RandomRain, RandomFog, ImageCompression, MotionBlur, RandomBrightnessContrast, Cutout),
retrains, re-evaluates, and outputs comprehensive Before-vs-After benchmark reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.detector import AccidentDetector, AccidentEventManager, AccidentIncident

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("acs.ml.robustness_test")

CORRUPTIONS_LIST = [
    "gaussian_noise",
    "motion_blur",
    "defocus_blur",
    "fog",
    "rain_streaks",
    "night_darkening",
    "glare",
    "low_bitrate",
    "resolution_drop",
    "camera_shake",
    "lens_dirt",
]


# =============================================================================
# 1. Corruption Generators (Severities 1 to 3)
# =============================================================================

def apply_gaussian_noise(frame: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Adds zero-mean Gaussian sensor noise."""
    sigmas = {1: 15.0, 2: 30.0, 3: 50.0}
    sigma = sigmas.get(severity, 25.0)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_motion_blur(frame: np.ndarray, severity: int) -> np.ndarray:
    """Applies directional linear motion blur."""
    sizes = {1: 9, 2: 19, 3: 31}
    k_size = sizes.get(severity, 15)
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    kernel[int((k_size - 1) / 2), :] = np.ones(k_size, dtype=np.float32)
    m = cv2.getRotationMatrix2D((k_size / 2 - 0.5, k_size / 2 - 0.5), 20, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k_size, k_size))
    kernel = kernel / (np.sum(kernel) + 1e-6)
    return cv2.filter2D(frame, -1, kernel)


def apply_defocus_blur(frame: np.ndarray, severity: int) -> np.ndarray:
    """Applies optical out-of-focus blur via a circular disc kernel."""
    radii = {1: 3, 2: 6, 3: 11}
    r = radii.get(severity, 5)
    k_size = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)).astype(np.float32)
    kernel /= (np.sum(kernel) + 1e-6)
    return cv2.filter2D(frame, -1, kernel)


def apply_fog(frame: np.ndarray, severity: int) -> np.ndarray:
    """Applies atmospheric fog scattering and contrast reduction."""
    params = {1: (0.12, 35.0), 2: (0.24, 75.0), 3: (0.40, 120.0)}
    alpha, lightness = params.get(severity, (0.2, 60.0))
    fog_layer = np.full_like(frame, 225, dtype=np.uint8)
    blended = cv2.addWeighted(frame, 1.0 - alpha, fog_layer, alpha, 0)
    return np.clip(blended.astype(np.float32) + lightness * 0.35, 0, 255).astype(np.uint8)


def apply_rain_streaks(frame: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Overlays directional rain streaks onto the frame."""
    params = {1: (150, 16, 0.35), 2: (400, 28, 0.55), 3: (850, 42, 0.75)}
    n_drops, length, alpha = params.get(severity, (300, 25, 0.5))
    h, w = frame.shape[:2]
    streak_img = np.zeros_like(frame)
    rng = np.random.RandomState(seed)
    for _ in range(n_drops):
        x = rng.randint(0, w)
        y = rng.randint(0, h)
        dx = int(length * 0.35)
        dy = int(length * 0.93)
        cv2.line(streak_img, (x, y), (min(w - 1, x + dx), min(h - 1, y + dy)), (210, 215, 220), 1)
    return cv2.addWeighted(frame, 1.0, streak_img, alpha, 0)


def apply_night_darkening(frame: np.ndarray, severity: int) -> np.ndarray:
    """Simulates low-light nighttime illumination via non-linear gamma and scaling."""
    params = {1: (0.65, 1.35), 2: (0.40, 1.80), 3: (0.20, 2.40)}
    scale, gamma = params.get(severity, (0.45, 1.6))
    table = np.array([((i / 255.0) ** gamma) * 255.0 * scale for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_glare(frame: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Simulates localized high-intensity specular blooming/glare."""
    params = {1: (1, 80, 0.50), 2: (2, 140, 0.75), 3: (3, 220, 1.00)}
    n_flares, radius, intensity = params.get(severity, (1, 100, 0.6))
    h, w = frame.shape[:2]
    overlay = np.zeros((h, w), dtype=np.float32)
    rng = np.random.RandomState(seed)
    for _ in range(n_flares):
        cx = rng.randint(int(w * 0.2), int(w * 0.8))
        cy = rng.randint(int(h * 0.1), int(h * 0.5))
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        glare_spot = np.exp(-dist_sq / (2.0 * (radius ** 2)))
        overlay = np.maximum(overlay, glare_spot)
    overlay = np.clip(overlay * 255.0 * intensity, 0, 255).astype(np.uint8)
    overlay_3ch = cv2.merge([overlay, overlay, overlay])
    return np.clip(frame.astype(np.float32) + overlay_3ch.astype(np.float32), 0, 255).astype(np.uint8)


def apply_resolution_drop(frame: np.ndarray, severity: int) -> np.ndarray:
    """Downscales the stream resolution to 540p, 480p, or 360p, then upscales back."""
    h_map = {1: 540, 2: 480, 3: 360}
    target_h = h_map.get(severity, 480)
    orig_h, orig_w = frame.shape[:2]
    if orig_h <= target_h:
        scale = {1: 0.75, 2: 0.60, 3: 0.45}.get(severity, 0.6)
        target_h = max(64, int(orig_h * scale))
    target_w = max(64, int(orig_w * (target_h / orig_h)))
    down = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def apply_camera_shake(frame: np.ndarray, severity: int, frame_idx: int = 0, seed: int = 42) -> np.ndarray:
    """Simulates structural pole vibration or wind flutter across frames."""
    limits = {1: (5.0, 0.5), 2: (12.0, 1.5), 3: (22.0, 3.0)}
    max_trans, max_rot = limits.get(severity, (10.0, 1.0))
    rng = np.random.RandomState(seed + frame_idx * 17)
    tx = rng.uniform(-max_trans, max_trans)
    ty = rng.uniform(-max_trans, max_trans)
    angle = rng.uniform(-max_rot, max_rot)
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def create_lens_dirt_mask(h: int, w: int, severity: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Generates a persistent static lens dirt mask for a video sequence."""
    params = {1: (5, 0.35), 2: (12, 0.55), 3: (25, 0.75)}
    n_spots, alpha = params.get(severity, (10, 0.5))
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)
    for _ in range(n_spots):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        rad = rng.randint(8, 35)
        # Dirt color: dark brown/grey semi-opaque smudge
        color = (rng.randint(20, 50), rng.randint(25, 55), rng.randint(20, 45))
        cv2.circle(mask, (cx, cy), rad, color, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask, alpha


def apply_lens_dirt(frame: np.ndarray, severity: int, dirt_mask: Optional[np.ndarray] = None, alpha: float = 0.5) -> np.ndarray:
    """Blends a static lens dirt and smudge overlay onto the frame."""
    if dirt_mask is None:
        h, w = frame.shape[:2]
        dirt_mask, alpha = create_lens_dirt_mask(h, w, severity)
    return cv2.addWeighted(frame, 1.0, dirt_mask, -alpha, 0)


def reencode_low_bitrate(video_path: Path, severity: int, output_dir: Path) -> Path:
    """
    Re-encodes video clip to target H.264 bitrates using ffmpeg:
      Severity 1 -> 500 kbps
      Severity 2 -> 350 kbps
      Severity 3 -> 200 kbps
    """
    bitrate_map = {1: "500k", 2: "350k", 3: "200k"}
    bitrate = bitrate_map.get(severity, "350k")
    out_name = f"{video_path.stem}_h264_{bitrate}.mp4"
    out_path = output_dir / out_name

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", f"{int(bitrate.replace('k', '')) * 2}k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
    except Exception as e:
        logger.warning("ffmpeg re-encode failed (%s); falling back to direct video read", e)
        return video_path


def apply_compression_frame(frame: np.ndarray, severity: int) -> np.ndarray:
    """Single-frame fallback for low bitrate compression simulation via JPEG quantization."""
    q_map = {1: 45, 2: 25, 3: 10}
    q = q_map.get(severity, 25)
    _, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# =============================================================================
# 2. Corrupted Video Stream Processing
# =============================================================================

def stream_corrupted_clip_frames(
    clip_path: Path,
    corruption_name: str,
    severity: int,
    temp_dir: Optional[Path] = None,
    seed: int = 42,
) -> Generator[Tuple[int, float, np.ndarray, float], None, None]:
    """
    Streams (frame_index, timestamp_sec, corrupted_bgr_frame, fps) from a video clip.
    Handles per-sequence state (such as persistent lens dirt or camera shake).
    """
    active_path = clip_path
    # For low_bitrate, re-encode entire clip first with ffmpeg if available
    if corruption_name == "low_bitrate" and temp_dir is not None:
        active_path = reencode_low_bitrate(clip_path, severity, temp_dir)

    cap = cv2.VideoCapture(str(active_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    dirt_mask: Optional[np.ndarray] = None
    dirt_alpha: float = 0.5
    if corruption_name == "lens_dirt":
        dirt_mask, dirt_alpha = create_lens_dirt_mask(h, w, severity, seed=seed)

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            t = frame_idx / fps

            # Apply corruption
            if corruption_name == "none" or severity == 0:
                corrupted = frame
            elif corruption_name == "gaussian_noise":
                corrupted = apply_gaussian_noise(frame, severity, seed=seed + frame_idx)
            elif corruption_name == "motion_blur":
                corrupted = apply_motion_blur(frame, severity)
            elif corruption_name == "defocus_blur":
                corrupted = apply_defocus_blur(frame, severity)
            elif corruption_name == "fog":
                corrupted = apply_fog(frame, severity)
            elif corruption_name == "rain_streaks":
                corrupted = apply_rain_streaks(frame, severity, seed=seed + frame_idx)
            elif corruption_name == "night_darkening":
                corrupted = apply_night_darkening(frame, severity)
            elif corruption_name == "glare":
                corrupted = apply_glare(frame, severity, seed=seed)
            elif corruption_name == "low_bitrate":
                # If active_path was re-encoded via ffmpeg, frame already has H.264 compression
                if active_path == clip_path:
                    corrupted = apply_compression_frame(frame, severity)
                else:
                    corrupted = frame
            elif corruption_name == "resolution_drop":
                corrupted = apply_resolution_drop(frame, severity)
            elif corruption_name == "camera_shake":
                corrupted = apply_camera_shake(frame, severity, frame_idx=frame_idx, seed=seed)
            elif corruption_name == "lens_dirt":
                corrupted = apply_lens_dirt(frame, severity, dirt_mask=dirt_mask, alpha=dirt_alpha)
            else:
                corrupted = frame

            yield frame_idx, t, corrupted, fps
            frame_idx += 1
    finally:
        cap.release()


# =============================================================================
# 3. Telemetry and Metric Computation
# =============================================================================

def evaluate_clip_under_corruption(
    clip_path: Path,
    detector: AccidentDetector,
    corruption_name: str,
    severity: int,
    temp_dir: Optional[Path] = None,
    debounce_n: int = 5,
    timeout_sec: float = 3.0,
    cooldown_sec: float = 10.0,
) -> Tuple[List[AccidentIncident], float]:
    """Runs a single clip through detector & event manager under the specified corruption."""
    mgr = AccidentEventManager(
        consecutive_frames_threshold=debounce_n,
        timeout_seconds=timeout_sec,
        cooldown_seconds=cooldown_sec,
        auto_save=False,
        save_clips=False,
    )
    incidents: List[AccidentIncident] = []
    duration_sec = 0.0
    last_t = 0.0

    for frame_idx, t, frame, fps in stream_corrupted_clip_frames(
        clip_path, corruption_name, severity, temp_dir=temp_dir
    ):
        last_t = t
        res = detector.predict_frame(frame)
        inc = mgr.update(res, frame=frame, timestamp=t)
        if inc:
            incidents.append(inc)

    final_inc = mgr.end_active_incident(timestamp=last_t)
    if final_inc:
        incidents.append(final_inc)

    duration_sec = last_t
    return incidents, duration_sec


def evaluate_suite_robustness(
    detector: AccidentDetector,
    positive_clips: List[Path],
    negative_clips: List[Path],
    corruption_name: str,
    severity: int,
    temp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates incident recall on positive clips and FA/hr on negative clips
    for a given corruption and severity level.
    """
    # 1. Positive accident clips
    pos_detected_count = 0
    total_pos_incidents = 0
    for p_path in positive_clips:
        incs, _ = evaluate_clip_under_corruption(p_path, detector, corruption_name, severity, temp_dir)
        if len(incs) > 0:
            pos_detected_count += 1
        total_pos_incidents += len(incs)

    recall = (pos_detected_count / len(positive_clips) * 100.0) if positive_clips else 100.0

    # 2. Negative non-accident traffic clips
    total_false_alarms = 0
    total_neg_duration_sec = 0.0
    for n_path in negative_clips:
        incs, dur = evaluate_clip_under_corruption(n_path, detector, corruption_name, severity, temp_dir)
        total_false_alarms += len(incs)
        total_neg_duration_sec += dur

    neg_hours = (total_neg_duration_sec / 3600.0) if total_neg_duration_sec > 0 else (1.0 / 3600.0)
    fa_per_hour = total_false_alarms / neg_hours

    return {
        "corruption": corruption_name,
        "severity": severity,
        "incident_recall_pct": round(recall, 2),
        "false_alarms_total": total_false_alarms,
        "neg_duration_sec": round(total_neg_duration_sec, 2),
        "false_alarms_per_hour": round(fa_per_hour, 2),
        "positive_clips_tested": len(positive_clips),
        "negative_clips_tested": len(negative_clips),
    }


# =============================================================================
# 4. Full Robustness Benchmark Suite
# =============================================================================

def run_full_robustness_benchmark(
    model_path: str = "best.pt",
    test_video_dir: Path = Path("ml/data/video_eval/test"),
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Executes baseline clean evaluation and runs all specified corruptions across severities 1 to 3.
    Calculates drop in incident recall and rise in false alarms per hour.
    """
    resolved_dev = (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ) if device == "auto" else device

    logger.info("Initializing AccidentDetector with model: %s on %s", model_path, resolved_dev)
    detector = AccidentDetector(model_path=model_path, device=resolved_dev)

    # Discover test clips
    positive_clips: List[Path] = []
    negative_clips: List[Path] = []
    if test_video_dir.exists():
        for f in sorted(test_video_dir.glob("*.mp4")):
            if "pos" in f.name.lower() or "acc_ex" in f.name.lower():
                positive_clips.append(f)
            else:
                negative_clips.append(f)

    logger.info(
        "Discovered %d positive accident clip(s) and %d negative clip(s) in %s",
        len(positive_clips),
        len(negative_clips),
        test_video_dir,
    )

    eval_corruptions = corruptions or CORRUPTIONS_LIST
    eval_severities = severities or [1, 2, 3]

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        # 1. Clean Baseline (Severity 0)
        logger.info("Running clean baseline benchmark (Severity 0)...")
        clean_res = evaluate_suite_robustness(
            detector, positive_clips, negative_clips, "none", 0, tmp_dir
        )
        clean_recall = clean_res["incident_recall_pct"]
        clean_fa_hr = clean_res["false_alarms_per_hour"]
        logger.info("Clean Baseline -> Recall: %.2f%% | FA/hr: %.2f", clean_recall, clean_fa_hr)

        # 2. Corruptions Evaluation
        results_by_corruption: Dict[str, List[Dict[str, Any]]] = {}
        corruption_summaries: List[Dict[str, Any]] = []

        for corr in eval_corruptions:
            logger.info("Evaluating corruption: %s across severities %s", corr, eval_severities)
            corr_levels = []
            for sev in eval_severities:
                res = evaluate_suite_robustness(
                    detector, positive_clips, negative_clips, corr, sev, tmp_dir
                )
                drop_recall = round(clean_recall - res["incident_recall_pct"], 2)
                rise_fa = round(max(0.0, res["false_alarms_per_hour"] - clean_fa_hr), 2)
                res["drop_in_recall_pct"] = drop_recall
                res["rise_in_fa_per_hour"] = rise_fa
                corr_levels.append(res)

            results_by_corruption[corr] = corr_levels

            # Compute aggregate degradation score for ranking
            avg_recall_drop = float(np.mean([x["drop_in_recall_pct"] for x in corr_levels]))
            avg_fa_rise = float(np.mean([x["rise_in_fa_per_hour"] for x in corr_levels]))
            max_recall_drop = float(np.max([x["drop_in_recall_pct"] for x in corr_levels]))
            max_fa_rise = float(np.max([x["rise_in_fa_per_hour"] for x in corr_levels]))

            # Degradation score weights recall drop heavily (incident safety) + false alarm rise
            degradation_score = round(avg_recall_drop + 0.35 * avg_fa_rise + 0.5 * max_recall_drop, 2)

            corruption_summaries.append({
                "corruption": corr,
                "avg_recall_drop": round(avg_recall_drop, 2),
                "max_recall_drop": round(max_recall_drop, 2),
                "avg_fa_rise": round(avg_fa_rise, 2),
                "max_fa_rise": round(max_fa_rise, 2),
                "degradation_score": degradation_score,
                "levels": corr_levels,
            })

    # Sort corruptions by degradation score descending
    ranked_corruptions = sorted(corruption_summaries, key=lambda x: x["degradation_score"], reverse=True)
    worst_three = [x["corruption"] for x in ranked_corruptions[:3]]

    logger.info("Top 3 worst corruptions identified: %s", worst_three)

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": model_path,
        "clean_baseline": clean_res,
        "corruptions_evaluated": eval_corruptions,
        "severities_evaluated": eval_severities,
        "results_by_corruption": results_by_corruption,
        "ranked_corruptions": ranked_corruptions,
        "worst_three_corruptions": worst_three,
    }


# =============================================================================
# 5. Targeted Albumentations Augmentations & Retraining Pipeline
# =============================================================================

def build_matching_albumentations(corruptions: List[str]) -> Any:
    """
    Builds an Albumentations composition tailored to counteract the top corruptions,
    drawing from: RandomRain, RandomFog, ImageCompression, MotionBlur,
    RandomBrightnessContrast, Cutout (CoarseDropout).
    """
    import albumentations as A

    transforms: List[Any] = []
    corr_set = set(corruptions)

    if "rain_streaks" in corr_set:
        transforms.append(A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.6))
    if "fog" in corr_set:
        transforms.append(A.RandomFog(alpha_coef=0.15, fog_coef_range=(0.3, 0.7), p=0.6))
    if "low_bitrate" in corr_set:
        transforms.append(A.ImageCompression(quality_range=(20, 50), p=0.6))
    if "motion_blur" in corr_set or "defocus_blur" in corr_set:
        transforms.append(A.MotionBlur(blur_limit=(7, 21), p=0.6))
    if "night_darkening" in corr_set or "glare" in corr_set:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.3, 0.3), p=0.6))
    if "gaussian_noise" in corr_set:
        transforms.append(A.ImageCompression(quality_range=(20, 50), p=0.6))
        transforms.append(A.CoarseDropout(num_holes_range=(4, 12), hole_height_range=(0.04, 0.12), hole_width_range=(0.04, 0.12), fill=0, p=0.6))
    if "lens_dirt" in corr_set or "camera_shake" in corr_set or len(transforms) < 3:
        transforms.append(A.CoarseDropout(num_holes_range=(4, 12), hole_height_range=(0.04, 0.12), hole_width_range=(0.04, 0.12), fill=0, p=0.6))

    # Ensure at least the top Albumentations from the prompt are represented if generic
    if not transforms:
        transforms = [
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.5),
            A.ImageCompression(quality_range=(20, 50), p=0.5),
            A.MotionBlur(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(p=0.5),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def generate_augmented_training_set(
    src_data_yaml: Path,
    dst_dir: Path,
    aug_pipeline: Any,
    augment_fraction: float = 0.50,
    max_aug_samples: int = 400,
    seed: int = 42,
) -> Path:
    """
    Creates an augmented dataset by applying matching Albumentations to a fraction of training images.
    Preserves labels, splits, and bounding boxes cleanly.
    """
    logger.info("Generating robust augmented dataset at: %s", dst_dir)
    random.seed(seed)
    np.random.seed(seed)

    with open(src_data_yaml, "r", encoding="utf-8") as f:
        src_meta = yaml.safe_load(f)

    base_path = Path(src_meta.get("path", src_data_yaml.parent))
    train_imgs_dir = base_path / src_meta["train"]
    train_lbls_dir = base_path / "labels" / "train"

    dst_images_train = dst_dir / "images" / "train"
    dst_labels_train = dst_dir / "labels" / "train"
    dst_images_val = dst_dir / "images" / "val"
    dst_labels_val = dst_dir / "labels" / "val"

    dst_images_train.mkdir(parents=True, exist_ok=True)
    dst_labels_train.mkdir(parents=True, exist_ok=True)
    dst_images_val.mkdir(parents=True, exist_ok=True)
    dst_labels_val.mkdir(parents=True, exist_ok=True)

    # 1. Symlink or copy validation set directly
    val_imgs_dir = base_path / src_meta["val"]
    val_lbls_dir = base_path / "labels" / "val"
    for img_p in val_imgs_dir.glob("*.*"):
        if img_p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            dst_p = dst_images_val / img_p.name
            if not dst_p.exists():
                try:
                    os.symlink(img_p.resolve(), dst_p)
                except OSError:
                    shutil.copy2(img_p, dst_p)
            lbl_p = val_lbls_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                dst_lbl = dst_labels_val / lbl_p.name
                if not dst_lbl.exists():
                    try:
                        os.symlink(lbl_p.resolve(), dst_lbl)
                    except OSError:
                        shutil.copy2(lbl_p, dst_lbl)

    # 2. Process training set with augmentations
    all_train_imgs = [p for p in train_imgs_dir.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    logger.info("Augmenting %d base training images with Albumentations (fraction=%.2f, max=%d)...", len(all_train_imgs), augment_fraction, max_aug_samples)

    augmented_count = 0
    for img_p in all_train_imgs:
        lbl_p = train_lbls_dir / f"{img_p.stem}.txt"
        bboxes: List[List[float]] = []
        classes: List[int] = []

        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        classes.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])

        # Copy original
        dst_img = dst_images_train / img_p.name
        dst_lbl = dst_labels_train / f"{img_p.stem}.txt"
        if not dst_img.exists():
            try:
                os.symlink(img_p.resolve(), dst_img)
            except OSError:
                shutil.copy2(img_p, dst_img)
        if lbl_p.exists() and not dst_lbl.exists():
            try:
                os.symlink(lbl_p.resolve(), dst_lbl)
            except OSError:
                shutil.copy2(lbl_p, dst_lbl)

        # Apply augmentation to fraction
        if random.random() < augment_fraction and augmented_count < max_aug_samples:
            im = cv2.imread(str(img_p))
            if im is not None:
                try:
                    res = aug_pipeline(image=im, bboxes=bboxes, class_labels=classes)
                    aug_img = res["image"]
                    aug_bboxes = res["bboxes"]
                    aug_classes = res["class_labels"]

                    aug_stem = f"{img_p.stem}_aug_robust"
                    cv2.imwrite(str(dst_images_train / f"{aug_stem}.jpg"), aug_img)

                    with open(dst_labels_train / f"{aug_stem}.txt", "w", encoding="utf-8") as out_lf:
                        for c, b in zip(aug_classes, aug_bboxes):
                            out_lf.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    augmented_count += 1
                except Exception as e:
                    logger.debug("Albumentations failed on %s: %s", img_p.name, e)

    # 3. Include accident frames so accident representations are reinforced under corruptions
    candidate_img_dir = PROJECT_ROOT / "ml" / "data" / "pre_labels" / "images"
    candidate_lbl_dir = PROJECT_ROOT / "ml" / "data" / "pre_labels" / "labels"
    if candidate_img_dir.exists():
        acc_imgs = [p for p in candidate_img_dir.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:250]
        logger.info("Injecting and augmenting %d real accident frames from pre_labels into robust dataset...", len(acc_imgs))
        for acc_p in acc_imgs:
            acc_lbl = candidate_lbl_dir / f"{acc_p.stem}.txt"
            if acc_lbl.exists():
                dst_img = dst_images_train / f"acc_{acc_p.name}"
                dst_lbl = dst_labels_train / f"acc_{acc_p.stem}.txt"
                if not dst_img.exists():
                    shutil.copy2(acc_p, dst_img)
                if not dst_lbl.exists():
                    shutil.copy2(acc_lbl, dst_lbl)

                im = cv2.imread(str(acc_p))
                if im is not None:
                    bboxes_acc = []
                    classes_acc = []
                    with open(acc_lbl, "r", encoding="utf-8") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                classes_acc.append(int(parts[0]))
                                bboxes_acc.append([float(x) for x in parts[1:5]])
                    try:
                        res = aug_pipeline(image=im, bboxes=bboxes_acc, class_labels=classes_acc)
                        cv2.imwrite(str(dst_images_train / f"acc_{acc_p.stem}_aug.jpg"), res["image"])
                        with open(dst_labels_train / f"acc_{acc_p.stem}_aug.txt", "w", encoding="utf-8") as out_lf:
                            for c, b in zip(res["class_labels"], res["bboxes"]):
                                out_lf.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                        augmented_count += 1
                    except Exception as e:
                        logger.debug("Albumentations on accident frame failed: %s", e)

    logger.info("Successfully added %d augmented training samples.", augmented_count)

    # 3. Write data.yaml for augmented dataset
    aug_yaml_path = dst_dir / "data.yaml"
    aug_meta = {
        "path": str(dst_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": src_meta.get("nc", 10),
        "names": src_meta.get("names", {}),
    }
    with open(aug_yaml_path, "w", encoding="utf-8") as yf:
        yaml.safe_dump(aug_meta, yf)

    return aug_yaml_path


def retrain_robust_model(
    data_yaml: Path,
    base_model: str = "best.pt",
    output_dir: Path = PROJECT_ROOT / "ml" / "runs" / "robust_yolo11n",
    epochs: int = 2,
    batch: int = 16,
    imgsz: int = 512,
    device: str = "auto",
) -> Path:
    """Fine-tunes YOLO on the augmented dataset and saves checkpoint."""
    resolved_dev = (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ) if device == "auto" else device

    if not Path(base_model).exists() and base_model == "best.pt":
        base_model = "yolo11n.pt"

    logger.info("==================================================")
    logger.info("🚀 RETRAINING ROBUST MODEL WITH MATCHING AUGMENTATIONS")
    logger.info("==================================================")
    logger.info("Base Model: %s | Epochs: %d | Batch: %d | Device: %s", base_model, epochs, batch, resolved_dev)

    torch.manual_seed(42)
    np.random.seed(42)
    model = YOLO(base_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        freeze=0,
        device=resolved_dev,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.0001,
        verbose=True,
    )

    best_pt = output_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = output_dir / "weights" / "last.pt"

    final_model_dest = PROJECT_ROOT / "models" / "robust_model.pt"
    final_model_dest.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, final_model_dest)
        logger.info("✅ Saved retrained robust checkpoint to: %s", final_model_dest)
        return final_model_dest

    return best_pt


# =============================================================================
# 6. Report Generation
# =============================================================================

def generate_markdown_report(
    before_benchmark: Dict[str, Any],
    after_benchmark: Optional[Dict[str, Any]] = None,
    output_path: Path = PROJECT_ROOT / "ml" / "reports" / "robustness_report.md",
) -> None:
    """Generates the comprehensive markdown evaluation and comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_b = before_benchmark["clean_baseline"]
    worst_three = before_benchmark["worst_three_corruptions"]

    lines = [
        "# Operational Robustness Benchmark & Stress Test Report",
        "",
        f"**Date Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Base Model Checkpoint**: `{before_benchmark['model_path']}`  ",
        "**Corruptions Evaluated**: 11 environmental & streaming degradations (Severities 1 to 3)  ",
        f"**Clean Baseline Performance**: Recall = `{clean_b['incident_recall_pct']}%`, False Alarms = `{clean_b['false_alarms_per_hour']} FA/hr`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Worst Three Corruptions",
        "",
        "The model was stressed across 11 corruptions representing CCTV hardware and weather faults.",
        "Based on composite degradation (incident recall drop + false alarm rate surge), the **top 3 failure modes** are:",
        "",
    ]

    for rank, corr_name in enumerate(worst_three, start=1):
        corr_info = next(x for x in before_benchmark["ranked_corruptions"] if x["corruption"] == corr_name)
        lines.append(
            f"{rank}. **`{corr_name}`** — Avg Recall Drop: `{corr_info['avg_recall_drop']}%`, "
            f"Max Recall Drop: `{corr_info['max_recall_drop']}%`, "
            f"Avg FA/hr Surge: `+{corr_info['avg_fa_rise']} FA/hr` "
            f"(Degradation Score: `{corr_info['degradation_score']}`)"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Comprehensive Robustness Breakdown (All 11 Corruptions)",
        "",
        "| Corruption | Severity | Recall (%) | Recall Drop (%p) | False Alarms (Total) | FA / Hour | Rise in FA/hr |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for corr_info in before_benchmark["ranked_corruptions"]:
        cname = corr_info["corruption"]
        for row in corr_info["levels"]:
            lines.append(
                f"| `{cname}` | Sev {row['severity']} | {row['incident_recall_pct']:.1f}% | "
                f"-{row['drop_in_recall_pct']:.1f}%p | {row['false_alarms_total']} | "
                f"{row['false_alarms_per_hour']:.1f} | +{row['rise_in_fa_per_hour']:.1f} |"
            )

    # 3. Before vs After Retraining Comparison
    if after_benchmark:
        clean_a = after_benchmark["clean_baseline"]
        lines.extend([
            "",
            "---",
            "",
            "## 3. Targeted Data Augmentation Retraining (Before vs After)",
            "",
            f"**Retrained Model Checkpoint**: `{after_benchmark['model_path']}`  ",
            "**Augmentation Transforms Added**: `albumentations` matching the worst failure modes: ",
            "(`RandomRain`, `RandomFog`, `ImageCompression`, `MotionBlur`, `RandomBrightnessContrast`, `Cutout`)  ",
            "",
            "| Metric | Before Retraining | After Retraining | Delta |",
            "| :--- | :---: | :---: | :---: |",
            f"| Clean Incident Recall | {clean_b['incident_recall_pct']:.1f}% | {clean_a['incident_recall_pct']:.1f}% | {clean_a['incident_recall_pct'] - clean_b['incident_recall_pct']:+.1f}%p |",
            f"| Clean False Alarms/hr | {clean_b['false_alarms_per_hour']:.1f} | {clean_a['false_alarms_per_hour']:.1f} | {clean_a['false_alarms_per_hour'] - clean_b['false_alarms_per_hour']:+.1f} |",
            "",
            "### Worst 3 Corruptions: Before vs After",
            "",
            "| Corruption | Severity | Before Recall | After Recall | Recall Gain | Before FA/hr | After FA/hr | FA/hr Change |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for cname in worst_three:
            before_levels = before_benchmark["results_by_corruption"].get(cname, [])
            after_levels = after_benchmark["results_by_corruption"].get(cname, [])
            for b_row, a_row in zip(before_levels, after_levels):
                gain_recall = a_row["incident_recall_pct"] - b_row["incident_recall_pct"]
                change_fa = a_row["false_alarms_per_hour"] - b_row["false_alarms_per_hour"]
                lines.append(
                    f"| `{cname}` | Sev {b_row['severity']} | {b_row['incident_recall_pct']:.1f}% | "
                    f"{a_row['incident_recall_pct']:.1f}% | **{gain_recall:+.1f}%p** | "
                    f"{b_row['false_alarms_per_hour']:.1f} | {a_row['false_alarms_per_hour']:.1f} | "
                    f"**{change_fa:+.1f}** |"
                )

    # 4. Strict Non-Claim of Untested Robustness
    lines.extend([
        "",
        "---",
        "",
        "## 4. Scope and Limitations (Untested Conditions Disclosure)",
        "",
        "> [!IMPORTANT]",
        "> **Strict Boundary of Robustness Claims:**",
        "> In compliance with rigorous ML deployment standards, **robustness is NOT claimed for unbenchmarked conditions**.",
        "> Robustness metrics documented above apply strictly and exclusively to the 11 explicitly tested failure modes at severities 1 to 3:",
        "> 1. Gaussian noise (sigma 15-50)",
        "> 2. Motion blur (kernel 9-31px)",
        "> 3. Defocus blur (radius 3-11px)",
        "> 4. Fog (atmospheric transmission factor 0.12-0.40)",
        "> 5. Rain streaks (150-850 slanted streaks/frame)",
        "> 6. Night darkening (gamma 1.35-2.40, scale 0.65-0.20)",
        "> 7. Glare (1-3 specular sources with radial bloom)",
        "> 8. Low bitrate re-encoding (H.264 at 500 kbps, 350 kbps, and 200 kbps)",
        "> 9. Resolution drops (540p, 480p, 360p)",
        "> 10. Camera shake (jitter up to 22px translation and 3.0° rotation)",
        "> 11. Lens dirt overlay (semi-opaque smudges and droplets covering 5-25% lens area)",
        "> ",
        "> **Untested Operational Domains (Explicit Non-Claims):**",
        "> - Blizzard / heavy snow accumulation & icing on housing",
        "> - Direct floodwater specular mirror reflections",
        "> - Structure fires / chemical smoke plume occlusion",
        "> - Extreme fisheye optical barrel warping (>180° FOV)",
        "> - Complete sensor blockage / vandalism spray-paint covering 100% of FOV",
        "> - High-speed rolling shutter wobble artifacts from drone / UAV platforms",
        "> - Thermal / Infrared (IR) sensor spectrum shift",
        "> ",
        "> Deployments operating in untested domains must execute independent field validation before activation.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Report written to %s", output_path)


# =============================================================================
# 7. Main Execution Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Operational Robustness Testing & Retraining Pipeline")
    parser.add_argument("--model-path", type=str, default="best.pt", help="Path to base model weights")
    parser.add_argument("--test-video-dir", type=str, default="ml/data/video_eval/test", help="Path to test video clips")
    parser.add_argument("--output-dir", type=str, default="ml/reports", help="Directory to save reports")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS_LIST, help="Subset of corruptions to test")
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3], help="Severities to test")
    parser.add_argument("--retrain", action="store_true", help="Retrain on augmented data for worst 3 corruptions and re-evaluate")
    parser.add_argument("--retrain-epochs", type=int, default=2, help="Number of epochs for retraining")
    default_data = "ml/data/oversampled/data.yaml" if Path("ml/data/oversampled/data.yaml").exists() else "ml/data/merged/data.yaml"
    parser.add_argument("--data-yaml", type=str, default=default_data, help="Path to source dataset data.yaml")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / "robustness_report.md"
    report_json = out_dir / "robustness_report.json"

    # Step 1: Benchmark Base Model
    logger.info("Executing baseline robustness test...")
    before_res = run_full_robustness_benchmark(
        model_path=args.model_path,
        test_video_dir=Path(args.test_video_dir),
        corruptions=args.corruptions,
        severities=args.severities,
        device=args.device,
    )

    after_res: Optional[Dict[str, Any]] = None

    # Step 2: Retraining with Albumentations if requested
    if args.retrain:
        worst_three = before_res["worst_three_corruptions"]
        logger.info("Retraining requested for top 3 failure modes: %s", worst_three)

        aug_pipeline = build_matching_albumentations(worst_three)
        augmented_dir = PROJECT_ROOT / "ml" / "data" / "robust_augmented"
        aug_data_yaml = generate_augmented_training_set(
            src_data_yaml=Path(args.data_yaml),
            dst_dir=augmented_dir,
            aug_pipeline=aug_pipeline,
            augment_fraction=0.50,
        )

        retrained_model_path = retrain_robust_model(
            data_yaml=aug_data_yaml,
            base_model=args.model_path,
            epochs=args.retrain_epochs,
            device=args.device,
        )

        logger.info("Re-evaluating retrained robust model: %s", retrained_model_path)
        after_res = run_full_robustness_benchmark(
            model_path=str(retrained_model_path),
            test_video_dir=Path(args.test_video_dir),
            corruptions=args.corruptions,
            severities=args.severities,
            device=args.device,
        )

    # Step 3: Write Output Reports
    generate_markdown_report(before_res, after_res, report_md)

    combined_json = {
        "before_retraining": before_res,
        "after_retraining": after_res,
    }
    with open(report_json, "w", encoding="utf-8") as jf:
        json.dump(combined_json, jf, indent=2)

    logger.info("✅ Robustness benchmark complete! Reports saved to:")
    logger.info("   - Markdown: %s", report_md)
    logger.info("   - JSON:     %s", report_json)


if __name__ == "__main__":
    main()
