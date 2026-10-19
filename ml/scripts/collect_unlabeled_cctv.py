#!/usr/bin/env python3
"""
Collect Unlabeled CCTV-Style Frames from Permitted Sources
---------------------------------------------------------
Curates diverse CCTV-style frames for teacher pseudo-labeling.
Permitted sources only:
1. Local public datasets (CADP CCTV, traffic surveillance, night traffic, weather variations).
2. User's own footage and simulation recordings (report_clips/*.mp4).
Strict Rules:
- NO scraping of external websites.
- NEVER touch or sample from ml/data/merged/images/val or test splits.
- Generates a source provenance manifest at ml/data/unlabeled_cctv/source_manifest.json.
"""

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Any
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_image_hash(image_path: Path) -> str:
    """Computes SHA256 of file content to prevent duplicate frames."""
    hasher = hashlib.sha256()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_frames_from_video(video_path: Path, output_dir: Path, step_frames: int = 25, max_frames: int = 30) -> List[Path]:
    """Extracts evenly spaced frames from a local video file."""
    extracted = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return extracted

    frame_idx = 0
    saved_count = 0
    vid_stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0 and saved_count < max_frames:
            # Check frame validity and minimal variance (skip pure black/corrupt frames)
            if frame.shape[0] >= 240 and frame.shape[1] >= 320 and frame.std() > 15:
                fname = f"user_clip_{vid_stem}_f{frame_idx:05d}.jpg"
                out_path = output_dir / fname
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                extracted.append(out_path)
                saved_count += 1
        frame_idx += 1

    cap.release()
    return extracted


def collect_unlabeled_frames(
    target_count: int = 1200,
    output_dir: Path = PROJECT_ROOT / "ml" / "data" / "unlabeled_cctv" / "frames",
    manifest_path: Path = PROJECT_ROOT / "ml" / "data" / "unlabeled_cctv" / "source_manifest.json",
    seed: int = 42
) -> Dict[str, Any]:
    """Collects diverse CCTV frames across multiple lighting, weather, and traffic conditions."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Protect test and val splits by recording their hashes
    frozen_hashes: Set[str] = set()
    for split in ["val", "test"]:
        split_dir = PROJECT_ROOT / "ml" / "data" / "merged" / "images" / split
        if split_dir.exists():
            for p in split_dir.glob("*.jpg"):
                frozen_hashes.add(get_image_hash(p))

    print(f"🔒 Guarding {len(frozen_hashes)} frozen val/test images against contamination.")

    sources = [
        {
            "name": "cadp_cctv",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "accident_candidates" / "cadp_cctv_frames",
            "quota": 45,
            "category": "cctv_traffic_accidents"
        },
        {
            "name": "cctv_accident_frames",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "accident_candidates" / "cctv_accident_frames",
            "quota": 300,
            "category": "cctv_collision_candidates"
        },
        {
            "name": "cctv_non_accident",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "hard_negatives" / "cctv_non_accident",
            "quota": 200,
            "category": "cctv_normal_baseline"
        },
        {
            "name": "rain_traffic",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "hard_negatives" / "rain_traffic",
            "quota": 200,
            "category": "cctv_rain_adverse_weather"
        },
        {
            "name": "traffic_jam",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "hard_negatives" / "traffic_jam",
            "quota": 150,
            "category": "cctv_congestion_close_passes"
        },
        {
            "name": "traffic_night",
            "path": PROJECT_ROOT / "ml" / "data" / "raw" / "hard_negatives" / "traffic_night",
            "quota": 250,
            "category": "cctv_night_low_light"
        },
    ]

    collected_records: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()

    # 1. Sample from permitted raw surveillance datasets
    for src in sources:
        src_path = src["path"]
        if not src_path.exists():
            continue
        all_imgs = list(src_path.rglob("*.jpg")) + list(src_path.rglob("*.png"))
        random.shuffle(all_imgs)

        sampled_from_src = 0
        for img_p in all_imgs:
            if sampled_from_src >= src["quota"]:
                break
            # Skip if image in val or test
            h = get_image_hash(img_p)
            if h in frozen_hashes or h in seen_hashes:
                continue

            seen_hashes.add(h)
            dst_name = f"unlabeled_{src['name']}_{img_p.name}"
            dst_path = output_dir / dst_name
            shutil.copy2(img_p, dst_path)

            collected_records.append({
                "filename": dst_name,
                "source": src["name"],
                "category": src["category"],
                "original_path": str(img_p.relative_to(PROJECT_ROOT)),
                "sha256": h
            })
            sampled_from_src += 1

        print(f"  + Collected {sampled_from_src} frames from {src['name']} ({src['category']})")

    # 2. Sample from user's own footage in report_clips/
    report_clips_dir = PROJECT_ROOT / "report_clips"
    if report_clips_dir.exists():
        video_clips = list(report_clips_dir.glob("*raw.mp4"))
        random.shuffle(video_clips)
        video_quota = 150
        video_collected = 0

        for v in video_clips[:15]:
            if video_collected >= video_quota:
                break
            extracted = extract_frames_from_video(v, output_dir, step_frames=20, max_frames=15)
            for f in extracted:
                h = get_image_hash(f)
                if h in frozen_hashes or h in seen_hashes:
                    f.unlink()
                    continue
                seen_hashes.add(h)
                collected_records.append({
                    "filename": f.name,
                    "source": "user_footage_report_clips",
                    "category": "cctv_user_simulation_recording",
                    "original_path": str(v.relative_to(PROJECT_ROOT)),
                    "sha256": h
                })
                video_collected += 1

        print(f"  + Extracted {video_collected} frames from user footage ({report_clips_dir.name})")

    manifest = {
        "curated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_unlabeled_frames": len(collected_records),
        "target_count": target_count,
        "permitted_sources": [s["name"] for s in sources] + ["user_footage_report_clips"],
        "category_breakdown": {},
        "frames": collected_records
    }

    # Count categories
    for r in collected_records:
        cat = r["category"]
        manifest["category_breakdown"][cat] = manifest["category_breakdown"].get(cat, 0) + 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Total Unlabeled Frames Collected: {len(collected_records)}")
    print(f"📄 Manifest written to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    collect_unlabeled_frames()
