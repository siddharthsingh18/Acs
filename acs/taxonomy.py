"""Load the canonical class taxonomy and explicitly decode legacy checkpoint IDs."""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSES_PATH = PROJECT_ROOT / "ml" / "configs" / "classes.yaml"
CHECKPOINT_MAPPINGS_PATH = PROJECT_ROOT / "ml" / "configs" / "checkpoint_class_mappings.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def load_classes() -> List[Dict[str, Any]]:
    """Return canonical classes ordered by their contiguous numeric IDs."""
    records = _read_yaml(CLASSES_PATH).get("classes")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No class records found in {CLASSES_PATH}")
    ordered = sorted(records, key=lambda item: item["id"])
    ids = [item["id"] for item in ordered]
    if ids != list(range(len(ordered))):
        raise ValueError("Class IDs must be contiguous and start at zero.")
    if any(item.get("group") not in {"accident", "normal"} for item in ordered):
        raise ValueError("Each class group must be 'accident' or 'normal'.")
    return ordered


def class_names() -> List[str]:
    return [item["name"] for item in load_classes()]


def class_records_by_name() -> Dict[str, Dict[str, Any]]:
    return {item["name"]: item for item in load_classes()}


def legacy_mapping() -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    doc = _read_yaml(CHECKPOINT_MAPPINGS_PATH).get("legacy_yolo_10_v1", {})
    source = doc.get("source_classes")
    mapping = doc.get("mapping")
    if not isinstance(source, list) or not isinstance(mapping, dict):
        raise ValueError("Invalid legacy checkpoint class mapping.")
    return source, {int(class_id): item for class_id, item in mapping.items()}


def decode_model_class(
    class_id: int,
    model_names: Mapping[int, str] | List[str],
) -> Tuple[int, str, Optional[str]]:
    """Decode a model output only when its ID/name pair has a declared mapping.

    Returns (canonical_id, canonical_name, incident_type). Unknown IDs or
    mismatched ID/name pairs fail rather than being interpreted positionally.
    """
    names = {int(key): str(value) for key, value in model_names.items()} if isinstance(model_names, Mapping) else {
        index: str(value) for index, value in enumerate(model_names)
    }
    if class_id not in names:
        raise ValueError(f"Checkpoint produced unknown class ID {class_id}.")
    output_name = names[class_id]
    canonical = load_classes()
    if class_id < len(canonical) and output_name == canonical[class_id]["name"]:
        return class_id, output_name, None

    legacy_names, mapping = legacy_mapping()
    if class_id < len(legacy_names) and output_name == legacy_names[class_id] and class_id in mapping:
        target_id = int(mapping[class_id]["target_id"])
        if target_id < 0 or target_id >= len(canonical):
            raise ValueError(f"Legacy class {class_id} maps outside the canonical taxonomy.")
        target_name = canonical[target_id]["name"]
        incident_type = mapping[class_id].get("incident_type")
        if incident_type and target_name != "accident":
            raise ValueError(f"Legacy subtype {incident_type} does not map to accident.")
        return target_id, target_name, incident_type

    raise ValueError(
        f"Checkpoint class order is unsupported: ID {class_id} is named {output_name!r}; "
        "add an explicit mapping before using these weights."
    )


def validate_model_taxonomy(model_names: Mapping[int, str] | List[str]) -> None:
    """Require canonical output order before using class-index-based metrics."""
    names = {int(key): str(value) for key, value in model_names.items()} if isinstance(model_names, Mapping) else {
        index: str(value) for index, value in enumerate(model_names)
    }
    expected = class_names()
    actual = [names.get(index) for index in range(len(names))]
    if actual != expected or len(names) != len(expected):
        raise ValueError(
            f"Checkpoint classes {actual!r} do not match the canonical order {expected!r}; "
            "class-index-based validation requires a canonical checkpoint. Legacy checkpoints "
            "can be decoded for inference but must not be evaluated with unremapped model.val() metrics."
        )
