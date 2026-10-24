import pytest

from ml.eval.metrics import (
    average_precision,
    bootstrap_camera_mean,
    class_metrics,
    clopper_pearson_interval,
    false_positive_frame_rate,
    match_predictions,
    poisson_false_alarm_upper_rate,
)


def test_hand_built_detection_case_has_exact_counts_and_ap():
    ground_truth = [(3, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        (3, 0.9, (0.0, 0.0, 10.0, 10.0)),
        (3, 0.8, (20.0, 20.0, 30.0, 30.0)),
    ]
    result = class_metrics(ground_truth, predictions, [3], iou_threshold=0.5)[3]
    assert result["ground_truth"] == 1
    assert result["predictions"] == 2
    assert result["true_positive"] == [1, 0]
    assert result["false_positive"] == [0, 1]
    assert result["ap"] == pytest.approx(1.0)


def test_duplicate_detections_and_iou_threshold_are_false_positives():
    ground_truth = [(3, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        (3, 0.9, (0.0, 0.0, 10.0, 10.0)),
        (3, 0.8, (0.0, 0.0, 10.0, 10.0)),
        (3, 0.7, (8.0, 8.0, 18.0, 18.0)),
    ]
    result = match_predictions(ground_truth, predictions, [3], iou_threshold=0.5)[3]
    assert result["true_positive"] == [1, 0, 0]
    assert result["false_positive"] == [0, 1, 1]
    assert average_precision([], [], 1) == 0.0


def test_empty_ground_truth_class_has_undefined_ap():
    result = class_metrics([], [(3, 0.9, (0.0, 0.0, 1.0, 1.0))], [3])[3]
    assert result["ground_truth"] == 0
    assert result["predictions"] == 1
    assert result["ap"] is None


def test_zero_false_alarm_rate_uses_rule_of_three():
    assert poisson_false_alarm_upper_rate(0, 20.0) == pytest.approx(2.995732 / 20.0)


def test_nonzero_false_alarm_bound_is_exact_one_sided_poisson_rate():
    assert poisson_false_alarm_upper_rate(1, 2.0) == pytest.approx(4.743865 / 2.0)


def test_frame_false_positive_rate_is_distinct_and_undefined_without_video_hours():
    assert false_positive_frame_rate(60, 1.5) == pytest.approx(40.0)
    assert false_positive_frame_rate(0, 2.0) == 0.0
    assert false_positive_frame_rate(0, 0.0) is None


def test_exact_recall_interval_and_minimum_all_successes():
    interval_16 = clopper_pearson_interval(16, 16)
    interval_17 = clopper_pearson_interval(17, 17)
    assert interval_16 is not None and interval_17 is not None
    assert interval_16[0] < 0.8
    assert interval_17[0] > 0.8
    assert clopper_pearson_interval(0, 0) is None


def test_bootstrap_resamples_cameras_not_frame_rows():
    first = bootstrap_camera_mean({"camera-a": 0.0, "camera-b": 1.0}, iterations=500, seed=4)
    second = bootstrap_camera_mean({"camera-a": 0.0, "camera-b": 1.0}, iterations=500, seed=4)
    assert first == second
    assert first is not None and first[0] == pytest.approx(0.5)
