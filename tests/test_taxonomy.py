from pathlib import Path

import yaml
import pytest

from acs.config import ACCIDENT_CLASSES, CLASS_NAMES, NORMAL_CLASSES
from acs.taxonomy import (
    CHECKPOINT_MAPPINGS_PATH,
    CLASSES_PATH,
    decode_model_class,
    legacy_mapping,
    load_classes,
    validate_model_taxonomy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_taxonomy_matches_canonical_yaml():
    records = load_classes()
    expected_names = [record["name"] for record in records]

    assert CLASS_NAMES == expected_names
    assert ACCIDENT_CLASSES == {record["name"] for record in records if record["group"] == "accident"}
    assert NORMAL_CLASSES == {record["name"] for record in records if record["group"] == "normal"}
    assert len({record["name"] for record in records}) == len(records)
    assert all(record.get("definition") for record in records)


def test_dataset_mapping_targets_match_canonical_class_names():
    records = load_classes()
    by_id = {record["id"]: record["name"] for record in records}
    class_map = yaml.safe_load((PROJECT_ROOT / "ml/configs/class_map.yaml").read_text(encoding="utf-8"))

    for dataset in class_map["datasets"].values():
        for mapping in dataset["class_mapping"].values():
            if mapping["action"] != "map":
                continue
            target_id = mapping["target_id"]
            assert target_id in by_id
            assert mapping["target_name"] == by_id[target_id]


def test_legacy_mapping_matches_exported_checkpoint_metadata():
    metadata_path = PROJECT_ROOT / "models" / "best_openvino_model" / "metadata.yaml"
    if not metadata_path.exists():
        metadata_path = PROJECT_ROOT / "best_openvino_model" / "metadata.yaml"
    if not metadata_path.exists():
        pytest.skip("Legacy OpenVINO checkpoint metadata is not present in this checkout.")

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    source_names, _ = legacy_mapping()
    exported_names = metadata["names"]
    if isinstance(exported_names, dict):
        exported_names = [exported_names[index] for index in sorted(exported_names, key=int)]

    assert exported_names == source_names


def test_legacy_prediction_decodes_to_canonical_accident_and_subtype():
    legacy_names, _ = legacy_mapping()

    canonical_id, canonical_name, incident_type = decode_model_class(6, legacy_names)

    assert (canonical_id, canonical_name, incident_type) == (3, "accident", "car_car_accident")


def test_class_index_metrics_reject_legacy_checkpoint_order():
    with pytest.raises(ValueError, match="class-index-based validation"):
        validate_model_taxonomy(legacy_mapping()[0])

    validate_model_taxonomy(CLASS_NAMES)


def test_decoder_rejects_unmapped_or_mismatched_class_order():
    with pytest.raises(ValueError, match="unsupported"):
        decode_model_class(0, ["car", "bike", "person", "accident"])

    with pytest.raises(ValueError, match="unknown class ID"):
        decode_model_class(10, ["bike", "car", "person", "accident"])


def test_review_queue_uses_backend_class_directory():
    review_page = (PROJECT_ROOT / "frontend/src/pages/ReviewQueuePage.tsx").read_text(encoding="utf-8")

    assert "CANONICAL_CLASSES" not in review_page
    assert "api.getConfig()" in review_page
    assert "systemConfig?.classes" in review_page


def test_runtime_sources_do_not_redeclare_legacy_detection_classes():
    legacy_names, _ = legacy_mapping()
    legacy_subtypes = set(legacy_names) - set(CLASS_NAMES)
    source_roots = [
        PROJECT_ROOT / "acs",
        PROJECT_ROOT / "backend/app",
        PROJECT_ROOT / "frontend/src",
        PROJECT_ROOT / "ml/scripts",
    ]
    allowed_subtype_metadata = PROJECT_ROOT / "frontend/src/lib/demoSimulation.ts"

    for source_root in source_roots:
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx", ".json"}:
                continue
            if ".test." in path.name or path == allowed_subtype_metadata:
                continue
            content = path.read_text(encoding="utf-8")
            assert not any(name in content for name in legacy_subtypes), path


def test_class_mapping_path_is_the_documented_single_source():
    assert CLASSES_PATH == PROJECT_ROOT / "ml/configs/classes.yaml"
    assert CHECKPOINT_MAPPINGS_PATH == PROJECT_ROOT / "ml/configs/checkpoint_class_mappings.yaml"
