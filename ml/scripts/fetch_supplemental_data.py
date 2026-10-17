#!/usr/bin/env python3
"""
Fetch Supplemental Data for ACS Model Hardening
----------------------------------------------
Downloads targeted datasets from Roboflow Universe using ROBOFLOW_API_KEY:
1. Hard Negative Domains:
   - cctv_non_accident: CCTV normal traffic / no accident (tammy/cctv-non-accident v2)
   - rain_traffic: Rain, wet asphalt, reflections & glare (weather-xduis/rain-kdcqi v1)
   - traffic_night: Night traffic, headlights, darkness (univ-kqors/traffic-night v1)
   - traffic_jam: Congestion, dense queues, braking (ballplayground/traffic-jam-ksie2 v3)
   - parked_cars: Roadside parked vehicles (jonathans-workspace-7jpdf/wrong-parked v2)
2. Accident Candidate Video Frames & Collision Datasets for Rare Class Pre-labeling:
   - cadp_cctv_frames: CADP CCTV crash video frames (yassine-pzpt7/cadp v1)
   - cctv_accident_frames: CCTV accident frame-by-frame (vehicle-accident-frame-by-frame/traffic-accident-nwmqo v5)
   - car_flip_object: Rollovers and obstacle impacts (latifa-fuirj/car-accident-jp29y v1)
   - moto_accident: Motorcycle collisions (deep-learning-xnbnl/accident-vcudo v17)
   - pedestrian_accident: Pedestrian vehicle collisions (sliit-kpo3y/pedestrian-accident-detection v3)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("Error: ROBOFLOW_API_KEY not found in environment.", file=sys.stderr)
    sys.exit(1)

rf = Roboflow(api_key=api_key)

# 1. Hard Negative Sources
NEG_PROJECTS = [
    {
        "name": "cctv_non_accident",
        "workspace": "tammy",
        "project": "cctv-non-accident",
        "version": 2,
        "target_dir": Path("ml/data/raw/hard_negatives/cctv_non_accident"),
    },
    {
        "name": "rain_traffic",
        "workspace": "weather-xduis",
        "project": "rain-kdcqi",
        "version": 1,
        "target_dir": Path("ml/data/raw/hard_negatives/rain_traffic"),
    },
    {
        "name": "traffic_night",
        "workspace": "univ-kqors",
        "project": "traffic-night",
        "version": 1,
        "target_dir": Path("ml/data/raw/hard_negatives/traffic_night"),
    },
    {
        "name": "traffic_jam",
        "workspace": "ballplayground",
        "project": "traffic-jam-ksie2",
        "version": 3,
        "target_dir": Path("ml/data/raw/hard_negatives/traffic_jam"),
    },
    {
        "name": "parked_cars",
        "workspace": "jonathans-workspace-7jpdf",
        "project": "wrong-parked",
        "version": 2,
        "target_dir": Path("ml/data/raw/hard_negatives/parked_cars"),
    },
]

# 2. Accident Candidate Sources for Auto-Labeling
ACC_PROJECTS = [
    {
        "name": "cadp_cctv_frames",
        "workspace": "yassine-pzpt7",
        "project": "cadp",
        "version": 1,
        "target_dir": Path("ml/data/raw/accident_candidates/cadp_cctv_frames"),
    },
    {
        "name": "cctv_accident_frames",
        "workspace": "vehicle-accident-frame-by-frame",
        "project": "traffic-accident-nwmqo",
        "version": 5,
        "target_dir": Path("ml/data/raw/accident_candidates/cctv_accident_frames"),
    },
    {
        "name": "car_flip_object",
        "workspace": "latifa-fuirj",
        "project": "car-accident-jp29y",
        "version": 1,
        "target_dir": Path("ml/data/raw/accident_candidates/car_flip_object"),
    },
    {
        "name": "moto_accident",
        "workspace": "deep-learning-xnbnl",
        "project": "accident-vcudo",
        "version": 17,
        "target_dir": Path("ml/data/raw/accident_candidates/moto_accident"),
    },
    {
        "name": "pedestrian_accident",
        "workspace": "sliit-kpo3y",
        "project": "pedestrian-accident-detection",
        "version": 3,
        "target_dir": Path("ml/data/raw/accident_candidates/pedestrian_accident"),
    },
]


def download_project(info: dict):
    target_dir = info["target_dir"]
    if target_dir.exists() and any(target_dir.rglob("*.jpg")):
        img_count = len(list(target_dir.rglob("*.jpg")))
        print(f"Skipping {info['name']}: already downloaded ({img_count} images in {target_dir})")
        return

    print(f"\nDownloading [{info['name']}] from {info['workspace']}/{info['project']} (v{info['version']})...")
    try:
        project = rf.workspace(info["workspace"]).project(info["project"])
        dataset = project.version(info["version"]).download(
            model_format="yolov11",
            location=str(target_dir.resolve()),
            overwrite=True
        )
        img_count = len(list(target_dir.rglob("*.jpg")))
        print(f"Successfully downloaded {info['name']}: {img_count} images in {target_dir}")
    except Exception as e:
        print(f"Error downloading {info['name']}: {e}", file=sys.stderr)


def main():
    print("==================================================================")
    print("Fetching Supplemental Datasets: Hard Negatives & Accident Footage")
    print("==================================================================")

    print("\n--- 1. Hard Negative Domains ---")
    for info in NEG_PROJECTS:
        download_project(info)

    print("\n--- 2. Accident Footage for Rare Class Auto-Labeling ---")
    for info in ACC_PROJECTS:
        download_project(info)

    print("\n==================================================================")
    print("Supplemental datasets download complete.")
    print("==================================================================")


if __name__ == "__main__":
    main()
