"""Small, deterministic metrics shared by image and video evaluation."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta, chi2


Box = tuple[float, float, float, float]
GroundTruth = tuple[int, Box] | tuple[str, int, Box]
Prediction = tuple[int, float, Box] | tuple[str, int, float, Box]


def _ground_truth_item(item: GroundTruth) -> tuple[str, int, Box]:
    if len(item) == 2:
        class_id, box = item
        return "", class_id, box
    image_id, class_id, box = item
    return image_id, class_id, box


def _prediction_item(item: Prediction) -> tuple[str, int, float, Box]:
    if len(item) == 3:
        class_id, confidence, box = item
        return "", class_id, confidence, box
    image_id, class_id, confidence, box = item
    return image_id, class_id, confidence, box


def box_iou(left: Box, right: Box) -> float:
    """IoU for xyxy boxes; degenerate boxes have zero overlap."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_predictions(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    class_ids: Iterable[int],
    iou_threshold: float = 0.50,
) -> dict[int, dict[str, object]]:
    """Greedy confidence-ranked matching; each GT may match at most one prediction."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1].")
    result = {}
    for class_id in class_ids:
        class_gt = [
            (image_id, box)
            for item in ground_truth
            for image_id, gt_class, box in (_ground_truth_item(item),)
            if gt_class == class_id
        ]
        class_predictions = sorted(
            (item for item in predictions if _prediction_item(item)[1] == class_id),
            key=lambda item: _prediction_item(item)[2],
            reverse=True,
        )
        matched_gt: set[int] = set()
        true_positive = []
        false_positive = []
        for prediction in class_predictions:
            image_id, _, _, prediction_box = _prediction_item(prediction)
            overlaps = [
                (box_iou(prediction_box, gt_box), index)
                for index, (gt_image_id, gt_box) in enumerate(class_gt)
                if index not in matched_gt and gt_image_id == image_id
            ]
            best_iou, best_index = max(overlaps, default=(0.0, -1))
            is_match = best_iou >= iou_threshold
            true_positive.append(int(is_match))
            false_positive.append(int(not is_match))
            if is_match:
                matched_gt.add(best_index)
        result[class_id] = {
            "ground_truth": len(class_gt),
            "predictions": len(class_predictions),
            "true_positive": true_positive,
            "false_positive": false_positive,
        }
    return result


def average_precision(true_positive: Sequence[int], false_positive: Sequence[int], gt_count: int) -> float | None:
    """Area under the interpolated precision-recall curve at the fixed score ordering."""
    if gt_count < 0 or len(true_positive) != len(false_positive):
        raise ValueError("Invalid GT count or mismatched TP/FP arrays.")
    if gt_count == 0:
        return None
    if not true_positive:
        return 0.0
    tp = np.cumsum(np.asarray(true_positive, dtype=float))
    fp = np.cumsum(np.asarray(false_positive, dtype=float))
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def class_metrics(
    ground_truth: Sequence[GroundTruth],
    predictions: Sequence[Prediction],
    class_ids: Iterable[int],
    iou_threshold: float = 0.50,
) -> dict[int, dict[str, object]]:
    matched = match_predictions(ground_truth, predictions, class_ids, iou_threshold)
    for metrics in matched.values():
        metrics["ap"] = average_precision(
            metrics["true_positive"], metrics["false_positive"], int(metrics["ground_truth"])
        )
    return matched


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Two-sided exact binomial interval; each independent event is one trial."""
    if trials < 0 or successes < 0 or successes > trials or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid binomial interval arguments.")
    if trials == 0:
        return None
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def poisson_false_alarm_upper_rate(
    false_alarms: int,
    camera_hours: float,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Poisson upper rate; zero events uses the 2.996/T rule of three."""
    if false_alarms < 0 or camera_hours <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("False alarms must be non-negative and camera_hours positive.")
    if false_alarms == 0:
        return float(-np.log(1.0 - confidence) / camera_hours)
    return float(chi2.ppf(confidence, 2 * (false_alarms + 1)) / (2 * camera_hours))


def false_positive_frame_rate(false_positive_frames: int, monitored_hours: float) -> float | None:
    """Frame-level error rate, intentionally distinct from incident false alarms/hour."""
    if false_positive_frames < 0 or monitored_hours < 0:
        raise ValueError("False-positive frames and monitored hours must be non-negative.")
    if monitored_hours == 0:
        return None
    return false_positive_frames / monitored_hours


def bootstrap_camera_mean(
    camera_values: dict[str, float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float] | None:
    """Percentile bootstrap over independent cameras/scenes, never frames."""
    if not camera_values:
        return None
    if iterations < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("iterations must be positive and confidence in (0, 1).")
    values = np.asarray(list(camera_values.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Camera-level values must be finite.")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(values.mean()), float(lower), float(upper)
