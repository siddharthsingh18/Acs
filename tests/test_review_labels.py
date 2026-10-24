from ml.scripts.review_labels import rank_image


def test_ranker_flags_unlabeled_vehicles_in_empty_label_images():
    result = rank_image(
        [],
        [(1, 0.93, (0.0, 0.0, 10.0, 10.0))],
        {3},
        confidence_floor=0.5,
    )
    assert result == (0.93, "possible_unlabeled_vehicle_or_road_user")


def test_ranker_flags_accident_labels_disagreed_with_by_checkpoint():
    result = rank_image(
        [(3, (0.0, 0.0, 10.0, 10.0))],
        [(1, 0.9, (0.0, 0.0, 10.0, 10.0))],
        {3},
        confidence_floor=0.5,
    )
    assert result == (0.9, "confident_other_class_overlaps_accident_label")
