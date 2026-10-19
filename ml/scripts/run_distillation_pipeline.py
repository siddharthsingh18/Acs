#!/usr/bin/env python3
"""
ACS Knowledge Distillation Master Pipeline Runner
-------------------------------------------------
Orchestrates the entire 6-step distillation workflow:
1. Teacher model training & evaluation on frozen test split.
2. CCTV frame collection from permitted local sources.
3. Pseudo-labeling with consistency checks (flip, scale, zero-shot collision).
4. Distillation dataset creation with pseudo-label cap <= 50%.
5. Student model (YOLO11n) training.
6. Comprehensive comparative evaluation against production baseline with bootstrap CIs.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.scripts.train_teacher import train_teacher
from ml.scripts.collect_unlabeled_cctv import collect_unlabeled_frames
from ml.scripts.pseudo_label import run_pseudo_labeling
from ml.scripts.build_distillation_dataset import build_distillation_dataset
from ml.scripts.train_student import train_student
from ml.scripts.evaluate_distillation import compare_models


def run_pipeline(skip_teacher_train: bool = False, skip_collection: bool = False):
    t_start = time.time()
    print("=" * 80)
    print("🚀 ACS MASTER KNOWLEDGE DISTILLATION WORKFLOW")
    print("=" * 80)

    # Step 1: Teacher Fine-Tuning
    teacher_ckpt = PROJECT_ROOT / "models" / "teacher_yolo11m.pt"
    if not skip_teacher_train or not teacher_ckpt.exists():
        print("\n--- STEP 1: TEACHER MODEL FINE-TUNING ---")
        train_teacher(epochs=2, batch=16, imgsz=640)
    else:
        print(f"\n--- STEP 1: Skipping Teacher Training (using existing {teacher_ckpt}) ---")

    # Step 2: CCTV Unlabeled Frame Collection
    manifest_unlabeled = PROJECT_ROOT / "ml" / "data" / "unlabeled_cctv" / "source_manifest.json"
    if not skip_collection or not manifest_unlabeled.exists():
        print("\n--- STEP 2: COLLECTING UNLABELED CCTV FRAMES ---")
        collect_unlabeled_frames()
    else:
        print(f"\n--- STEP 2: Skipping Collection (using existing manifest {manifest_unlabeled}) ---")

    # Step 3: Teacher Pseudo-Labeling with Consistency Checks
    print("\n--- STEP 3 & 4: TEACHER PSEUDO-LABELING & CONTACT SHEETS ---")
    run_pseudo_labeling(teacher_weights=teacher_ckpt)

    # Step 5: Build Distillation Dataset (<= 50% cap)
    print("\n--- STEP 5: BUILDING DISTILLATION TRAINING DATASET ---")
    build_distillation_dataset(max_pseudo_share=0.50)

    # Step 6: Train Student (YOLO11n)
    print("\n--- STEP 6: TRAINING STUDENT MODEL (YOLO11n) ---")
    student_ckpt = train_student(epochs=2, batch=16, imgsz=640)

    # Step 7: Comprehensive Comparative Evaluation
    print("\n--- STEP 7: EVALUATING STUDENT VS PRODUCTION BASELINE ---")
    compare_models(student_path=student_ckpt)

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "=" * 80)
    print(f"🎉 DISTILLATION WORKFLOW COMPLETE IN {elapsed}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()
    run_pipeline(skip_teacher_train=args.skip_teacher, skip_collection=args.skip_collection)
