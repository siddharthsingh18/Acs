"""
acs.privacy
Lightweight privacy protection module for CCTV footage.
Detects and blurs human faces and vehicle license plates on stored keyframes,
video clips, and PDF reports. Supports an access-controlled vault for unredacted originals.
"""
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from acs.config import config

logger = logging.getLogger("acs.privacy")


def _box_area(box: Tuple[int, int, int, int, str]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_iou(box_a: Tuple[int, int, int, int, str], box_b: Tuple[int, int, int, int, str]) -> float:
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = _box_area(box_a)
    areaB = _box_area(box_b)
    union = areaA + areaB - inter
    return inter / float(union) if union > 0 else 0.0


def merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int, str]], iou_threshold: float = 0.25
) -> List[Tuple[int, int, int, int, str]]:
    """Merges overlapping bounding boxes to avoid redundant blurring passes."""
    if not boxes:
        return []

    merged: List[Tuple[int, int, int, int, str]] = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2, label = boxes[i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if not used[j] and _box_iou((x1, y1, x2, y2, label), boxes[j]) > iou_threshold:
                # Merge into bounding union
                x1 = min(x1, boxes[j][0])
                y1 = min(y1, boxes[j][1])
                x2 = max(x2, boxes[j][2])
                y2 = max(y2, boxes[j][3])
                used[j] = True

        merged.append((x1, y1, x2, y2, label))

    return merged


def detect_faces(
    frame: np.ndarray, detected_boxes: Optional[List[Union[dict, Any]]] = None
) -> List[Tuple[int, int, int, int, str]]:
    """Detects facial regions using person bounding box guidance and skin-chrominance filtering.

    Args:
        frame: BGR image array (uint8).
        detected_boxes: Optional list of upstream YOLO detections.

    Returns:
        List of (x1, y1, x2, y2, 'face') coordinates.
    """
    h, w = frame.shape[:2]
    faces: List[Tuple[int, int, int, int, str]] = []

    # 1. Person-guided head/face region extraction
    if detected_boxes:
        for box in detected_boxes:
            c_name = getattr(box, "class_name", None) or (box.get("class_name") if isinstance(box, dict) else "")
            c_name = str(c_name).lower()
            if any(term in c_name for term in ("person", "pedestrian", "cyclist", "rider")):
                coords = getattr(box, "xyxy", None) or (box.get("xyxy") if isinstance(box, dict) else None)
                if coords and len(coords) >= 4:
                    bx1, by1, bx2, by2 = [int(v) for v in coords[:4]]
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)
                    # Head/face is typically top 28% of the person box
                    face_y2 = min(h, by1 + int(bh * 0.32))
                    faces.append((max(0, bx1), max(0, by1), min(w, bx2), face_y2, "face"))

    # 2. Skin tone chrominance filter (YCrCb: Cr in [133, 173], Cb in [77, 127])
    # Fast heuristic to find un-annotated faces in frame
    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            aspect = ch / float(cw) if cw > 0 else 0
            area = cw * ch
            # Faces have roughly oval aspect ratio (0.8 to 2.2) and realistic area
            min_area = max(200, int((w * h) * 0.0003))
            max_area = int((w * h) * 0.15)
            if 0.8 <= aspect <= 2.2 and min_area <= area <= max_area:
                faces.append((cx, cy, cx + cw, cy + ch, "face"))
    except Exception as exc:
        logger.debug("Skin tone face heuristic error: %s", exc)

    return faces


def detect_license_plates(
    frame: np.ndarray, detected_boxes: Optional[List[Union[dict, Any]]] = None
) -> List[Tuple[int, int, int, int, str]]:
    """Detects vehicle license plate regions using edge morphology and vehicle bounding box guidance.

    Args:
        frame: BGR image array (uint8).
        detected_boxes: Optional list of upstream YOLO detections.

    Returns:
        List of (x1, y1, x2, y2, 'license_plate') coordinates.
    """
    h, w = frame.shape[:2]
    plates: List[Tuple[int, int, int, int, str]] = []

    # 1. Vehicle-guided plate region extraction (lower-center 30% of vehicle)
    if detected_boxes:
        for box in detected_boxes:
            c_name = getattr(box, "class_name", None) or (box.get("class_name") if isinstance(box, dict) else "")
            c_name = str(c_name).lower()
            if any(term in c_name for term in ("car", "truck", "bus", "bike", "motorcycle", "vehicle")):
                coords = getattr(box, "xyxy", None) or (box.get("xyxy") if isinstance(box, dict) else None)
                if coords and len(coords) >= 4:
                    bx1, by1, bx2, by2 = [int(v) for v in coords[:4]]
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)
                    # Plate resides in the lower third
                    plate_x1 = max(0, bx1 + int(bw * 0.15))
                    plate_x2 = min(w, bx2 - int(bw * 0.15))
                    plate_y1 = max(0, by1 + int(bh * 0.65))
                    plate_y2 = min(h, by2)
                    if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                        plates.append((plate_x1, plate_y1, plate_x2, plate_y2, "license_plate"))

    # 2. High-contrast morphological edge detector for plate characters
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            aspect = cw / float(ch) if ch > 0 else 0
            area = cw * ch
            # License plates have rectangular aspect ratios (1.8 to 5.5)
            min_area = max(250, int((w * h) * 0.0004))
            max_area = int((w * h) * 0.08)
            if 1.8 <= aspect <= 5.5 and min_area <= area <= max_area:
                plates.append((cx, cy, cx + cw, cy + ch, "license_plate"))
    except Exception as exc:
        logger.debug("Morphological plate detector error: %s", exc)

    return plates


def detect_sensitive_regions(
    frame: np.ndarray,
    detected_boxes: Optional[List[Union[dict, Any]]] = None,
    blur_faces: bool = True,
    blur_plates: bool = True,
) -> List[Tuple[int, int, int, int, str]]:
    """Identifies all sensitive regions (faces and license plates) within an image frame.

    Args:
        frame: BGR image array.
        detected_boxes: Upstream YOLO detections.
        blur_faces: Whether to detect faces.
        blur_plates: Whether to detect license plates.

    Returns:
        List of non-overlapping (x1, y1, x2, y2, label) tuples.
    """
    candidates: List[Tuple[int, int, int, int, str]] = []

    if blur_faces:
        candidates.extend(detect_faces(frame, detected_boxes=detected_boxes))
    if blur_plates:
        candidates.extend(detect_license_plates(frame, detected_boxes=detected_boxes))

    return merge_overlapping_boxes(candidates, iou_threshold=0.25)


def apply_privacy_blur(
    frame: np.ndarray,
    regions: Optional[List[Tuple[int, int, int, int, str]]] = None,
    detected_boxes: Optional[List[Union[dict, Any]]] = None,
    blur_radius: int = 41,
    padding_pct: float = 0.12,
) -> np.ndarray:
    """Applies heavy Gaussian blurring to privacy-sensitive regions.

    Args:
        frame: BGR image array (uint8).
        regions: Optional pre-computed regions to blur. If None, detected automatically.
        detected_boxes: Optional YOLO detection boxes for guided extraction.
        blur_radius: Gaussian kernel size (must be odd).
        padding_pct: Percentage expansion around detected bounding box.

    Returns:
        np.ndarray: Privacy-protected frame copy.
    """
    if frame is None or frame.size == 0:
        return frame

    blurred_frame = frame.copy()
    h, w = frame.shape[:2]

    # Ensure odd kernel size
    ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    ksize = max(15, min(ksize, 99))

    if regions is None:
        regions = detect_sensitive_regions(frame, detected_boxes=detected_boxes)

    for x1, y1, x2, y2, _ in regions:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * padding_pct)
        pad_y = int(bh * padding_pct)

        # Expand box coordinates with boundaries clamped
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)

        if rx2 > rx1 and ry2 > ry1:
            roi = blurred_frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                blurred_frame[ry1:ry2, rx1:rx2] = blurred_roi

    return blurred_frame


def protect_keyframe(
    keyframe: np.ndarray,
    incident_id: str,
    output_path: str,
    detected_boxes: Optional[List[Union[dict, Any]]] = None,
    preserve_unblurred: bool = True,
    unredacted_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Saves privacy-blurred keyframe and optionally vaults the unredacted original.

    Args:
        keyframe: Input BGR keyframe array.
        incident_id: Incident identifier.
        output_path: Target path for the public blurred keyframe.
        detected_boxes: Optional YOLO bounding boxes.
        preserve_unblurred: Whether to keep original unblurred frame in vault.
        unredacted_dir: Vault root directory (defaults to config.privacy_unredacted_dir).

    Returns:
        Tuple of (blurred_image_path, unredacted_vault_path_or_None).
    """
    vault_dir = unredacted_dir or getattr(config, "privacy_unredacted_dir", "media/unredacted")
    unredacted_path = None

    # 1. Vault the unblurred original if requested
    if preserve_unblurred and keyframe is not None and keyframe.size > 0:
        kf_vault_dir = os.path.join(vault_dir, "keyframes")
        os.makedirs(kf_vault_dir, exist_ok=True)
        unredacted_path = os.path.join(kf_vault_dir, f"{incident_id}_unredacted.jpg")
        cv2.imwrite(unredacted_path, keyframe)

    # 2. Redact and save the public keyframe
    blurred = apply_privacy_blur(keyframe, detected_boxes=detected_boxes)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, blurred)

    return output_path, unredacted_path


def protect_video_clip(
    src_clip_path: str,
    dest_clip_path: str,
    incident_id: str,
    preserve_unblurred: bool = True,
    unredacted_dir: Optional[str] = None,
    fps: float = 30.0,
) -> Tuple[str, Optional[str]]:
    """Applies frame-by-frame privacy blur to video clip and optionally vaults unredacted file.

    Args:
        src_clip_path: Path to source H.264 video.
        dest_clip_path: Destination path for blurred clip.
        incident_id: Incident ID.
        preserve_unblurred: Whether to preserve original in vault.
        unredacted_dir: Vault directory.
        fps: Playback frames per second.

    Returns:
        Tuple of (blurred_clip_path, unredacted_clip_path_or_None).
    """
    if not os.path.exists(src_clip_path):
        return dest_clip_path, None

    vault_dir = unredacted_dir or getattr(config, "privacy_unredacted_dir", "media/unredacted")
    unredacted_path = None

    # 1. Vault original unredacted clip
    if preserve_unblurred:
        clip_vault_dir = os.path.join(vault_dir, "clips")
        os.makedirs(clip_vault_dir, exist_ok=True)
        unredacted_path = os.path.join(clip_vault_dir, f"{incident_id}_unredacted.mp4")
        shutil.copy2(src_clip_path, unredacted_path)

    # 2. Read frames, blur each, and encode to dest_clip_path
    cap = cv2.VideoCapture(src_clip_path)
    if not cap.isOpened():
        return dest_clip_path, unredacted_path

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        blurred_f = apply_privacy_blur(frame)
        frames.append(blurred_f)
    cap.release()

    if not frames:
        return dest_clip_path, unredacted_path

    h, w = frames[0].shape[:2]
    out_dir = os.path.dirname(dest_clip_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Encode with ffmpeg if available, otherwise OpenCV
    encoded = False
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            dest_clip_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(f.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=30)
        if proc.returncode == 0 and os.path.exists(dest_clip_path) and os.path.getsize(dest_clip_path) > 0:
            encoded = True
    except Exception as exc:
        logger.debug("ffmpeg privacy encoding fallback: %s", exc)

    if not encoded:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dest_clip_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    return dest_clip_path, unredacted_path
