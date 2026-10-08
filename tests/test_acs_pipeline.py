"""
tests/test_acs_pipeline.py
Pytest test suite covering:
1. Class filtering (only accident classes trigger; normal traffic classes do not)
2. Incident debounce (N consecutive frames required) & cooldown (10s delay enforced)
3. CSV reporter read/write (report generation, update, round-trip parsing, master log)
4. llm.py fallback when the API key is missing (with the network strictly mocked)
5. i18n key parity and placeholder parity between English and Arabic catalogs
"""
import os
import re
import socket
import tempfile
import urllib.request
from typing import Generator
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import requests

from acs.config import ACCIDENT_CLASSES, CLASS_NAMES
from acs.detector import (
    AccidentEventManager,
    DetectionBox,
    DetectionResult,
)
from acs.i18n import CLASS_TRANSLATIONS, UI_STRINGS, t, translate_class_name
from acs.llm import DispatchReportModel, generate_dispatch_report
from acs.reporter import (
    list_reports,
    read_incidents_csv,
    read_report_csv,
    save_keyframe,
    save_report_csv,
    update_report_description,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_dirs() -> Generator[tuple[str, str, str], None, None]:
    """Provides temporary directories for images, reports, and scratch files."""
    with tempfile.TemporaryDirectory() as base_tmp:
        images_dir = os.path.join(base_tmp, "report_images")
        reports_dir = os.path.join(base_tmp, "reports")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        yield base_tmp, images_dir, reports_dir


@pytest.fixture
def dummy_image_file(temp_dirs: tuple[str, str, str]) -> str:
    """Creates a temporary test JPEG image on disk and returns its path."""
    _, images_dir, _ = temp_dirs
    img_path = os.path.join(images_dir, "test_peak_frame.jpg")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.imwrite(img_path, frame)
    return img_path


def make_detection_result(
    class_name: str,
    confidence: float = 0.88,
    is_accident: bool = False,
) -> DetectionResult:
    """Helper creating a DetectionResult with given class and accident flag."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = DetectionBox(
        bbox=(10.0, 10.0, 60.0, 60.0),
        confidence=confidence,
        class_id=0,
        class_name=class_name,
        is_accident=is_accident,
        track_id=1,
    )
    return DetectionResult(
        annotated_frame=frame,
        detections=[box],
        accident_detected=is_accident,
        detected_classes={class_name},
        accident_classes={class_name} if is_accident else set(),
    )


# -----------------------------------------------------------------------------
# 1. Class Filtering Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("accident_class", sorted(list(ACCIDENT_CLASSES)))
def test_class_filtering_only_accident_classes_trigger(
    temp_dirs: tuple[str, str, str],
    accident_class: str,
) -> None:
    """All classes in ACCIDENT_CLASSES must trigger an incident after N consecutive frames."""
    _, images_dir, reports_dir = temp_dirs
    manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        timeout_seconds=2.0,
        cooldown_seconds=5.0,
        confidence_threshold=0.5,
        output_dir=images_dir,
        reports_dir=reports_dir,
        auto_save=False,
    )

    # Feed 5 consecutive frames of the accident class
    for i in range(5):
        t_sec = i * 0.1
        det = make_detection_result(class_name=accident_class, confidence=0.90, is_accident=True)
        manager.update(det, timestamp=t_sec)

    assert manager.is_incident_active is True
    assert manager.current_incident is not None
    assert manager.current_incident.class_name == accident_class


@pytest.mark.parametrize("normal_class", ["car", "person", "bike", "traffic_light", "truck"])
def test_class_filtering_non_accident_classes_do_not_trigger(
    temp_dirs: tuple[str, str, str],
    normal_class: str,
) -> None:
    """Non-accident traffic classes must NEVER trigger an incident even with 20 frames."""
    _, images_dir, reports_dir = temp_dirs
    manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        confidence_threshold=0.5,
        output_dir=images_dir,
        reports_dir=reports_dir,
        auto_save=False,
    )

    # Feed 20 consecutive frames of normal traffic detection
    for i in range(20):
        t_sec = i * 0.1
        det = make_detection_result(class_name=normal_class, confidence=0.99, is_accident=False)
        inc = manager.update(det, timestamp=t_sec)
        assert inc is None

    assert manager.is_incident_active is False
    assert len(manager.incidents) == 0


def test_class_filtering_mixed_frame_triggers_on_accident(
    temp_dirs: tuple[str, str, str],
) -> None:
    """When a frame contains both normal cars and an accident, accident must trigger."""
    _, images_dir, reports_dir = temp_dirs
    manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        output_dir=images_dir,
        reports_dir=reports_dir,
        auto_save=False,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    normal_box = DetectionBox(
        bbox=(0.0, 0.0, 20.0, 20.0),
        confidence=0.95,
        class_id=3,
        class_name="car",
        is_accident=False,
    )
    accident_box = DetectionBox(
        bbox=(30.0, 30.0, 80.0, 80.0),
        confidence=0.89,
        class_id=5,
        class_name="car_car_accident",
        is_accident=True,
    )

    mixed_result = DetectionResult(
        annotated_frame=frame,
        detections=[normal_box, accident_box],
        accident_detected=True,
        detected_classes={"car", "car_car_accident"},
        accident_classes={"car_car_accident"},
    )

    for i in range(5):
        manager.update(mixed_result, timestamp=i * 0.1)

    assert manager.is_incident_active is True
    assert manager.current_incident.class_name == "car_car_accident"


# -----------------------------------------------------------------------------
# 2. Incident Debounce and Cooldown Tests
# -----------------------------------------------------------------------------
def test_incident_debounce_requires_n_consecutive_frames(
    temp_dirs: tuple[str, str, str],
) -> None:
    """Debounce: flickering detections (< N consecutive frames) must be rejected."""
    _, images_dir, reports_dir = temp_dirs
    manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        timeout_seconds=3.0,
        output_dir=images_dir,
        reports_dir=reports_dir,
        auto_save=False,
    )

    accident_det = make_detection_result("car_car_accident", confidence=0.85, is_accident=True)
    quiet_det = make_detection_result("car", confidence=0.90, is_accident=False)

    # 1. Feed 4 accident frames (1 short of 5 threshold)
    for i in range(4):
        manager.update(accident_det, timestamp=i * 0.1)
    assert manager.is_incident_active is False

    # 2. Feed 1 quiet frame breaking the streak
    manager.update(quiet_det, timestamp=0.4)
    assert manager.is_incident_active is False

    # 3. Feed another 4 accident frames: still no incident
    for i in range(4):
        manager.update(accident_det, timestamp=0.5 + (i * 0.1))
    assert manager.is_incident_active is False

    # 4. Feed 5th consecutive accident frame: incident starts immediately
    manager.update(accident_det, timestamp=0.9)
    assert manager.is_incident_active is True


def test_incident_cooldown_enforcement(
    temp_dirs: tuple[str, str, str],
) -> None:
    """Cooldown: new incidents cannot start during the 10-second cooldown period."""
    _, images_dir, reports_dir = temp_dirs
    cooldown = 10.0
    timeout = 3.0
    manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        timeout_seconds=timeout,
        cooldown_seconds=cooldown,
        output_dir=images_dir,
        reports_dir=reports_dir,
        auto_save=False,
    )

    accident_det = make_detection_result("car_car_accident", confidence=0.88, is_accident=True)
    quiet_det = make_detection_result("car", confidence=0.90, is_accident=False)

    # 1. Start incident 1 (frames 0 to 4: t=0.0 to 0.4s)
    for i in range(5):
        manager.update(accident_det, timestamp=i * 0.1)
    assert manager.is_incident_active is True

    # 2. Quiet period to close incident 1 (exceed timeout 3.0s)
    closed_incident = None
    for i in range(5, 45):
        t_sec = i * 0.1
        res = manager.update(quiet_det, timestamp=t_sec)
        if res:
            closed_incident = res
    assert closed_incident is not None
    assert manager.is_incident_active is False
    assert len(manager.incidents) == 1

    end_time = closed_incident.end_time

    # 3. Feed accident detections DURING COOLDOWN (e.g., t = end_time + 4.0s < 10.0s)
    during_cooldown_start = end_time + 4.0
    for i in range(10):
        t_sec = during_cooldown_start + (i * 0.1)
        res = manager.update(accident_det, timestamp=t_sec)
        assert res is None
    # Incident must NOT start
    assert manager.is_incident_active is False
    assert len(manager.incidents) == 1

    # 4. Feed accident detections AFTER COOLDOWN EXPIRES (t = end_time + 11.0s > 10.0s)
    after_cooldown_start = end_time + 11.0
    for i in range(5):
        t_sec = after_cooldown_start + (i * 0.1)
        manager.update(accident_det, timestamp=t_sec)

    # Second incident must successfully trigger
    assert manager.is_incident_active is True
    active_inc = manager.current_incident
    assert active_inc is not None
    assert active_inc.incident_id != closed_incident.incident_id


# -----------------------------------------------------------------------------
# 3. CSV Reporter Read/Write Tests
# -----------------------------------------------------------------------------
def test_csv_reporter_save_and_read_report(
    temp_dirs: tuple[str, str, str],
) -> None:
    """save_report_csv and read_report_csv must write and read report data accurately."""
    _, images_dir, reports_dir = temp_dirs

    image_paths = [
        os.path.join(images_dir, "frame_01.jpg"),
        os.path.join(images_dir, "frame_02.jpg"),
    ]
    for p in image_paths:
        cv2.imwrite(p, np.zeros((50, 50, 3), dtype=np.uint8))

    filename, full_path = save_report_csv(
        class_name="car_car_accident",
        image_paths=image_paths,
        description="Head-on collision at intersection",
        confidence=0.94,
        reports_dir=reports_dir,
        timestamp="2026-10-08_12-00-00",
    )

    assert os.path.exists(full_path)
    assert filename.startswith("report_2026-10-08_12-00-00")

    # Read back CSV
    df = read_report_csv(full_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["Class"].iloc[0] == "car_car_accident"
    assert df["Description"].iloc[0] == "Head-on collision at intersection"
    assert pytest.approx(df["Confidence"].iloc[0], 0.01) == 0.94
    assert df["Images"].iloc[0] == ";".join(image_paths)


def test_csv_reporter_update_description(
    temp_dirs: tuple[str, str, str],
) -> None:
    """update_report_description must update existing report CSV in-place."""
    _, _, reports_dir = temp_dirs
    _, full_path = save_report_csv(
        class_name="bike_person_accident",
        image_paths=[],
        description="Initial automated label",
        reports_dir=reports_dir,
    )

    success = update_report_description(full_path, "Updated multimodal assessment")
    assert success is True

    df = read_report_csv(full_path)
    assert df["Description"].iloc[0] == "Updated multimodal assessment"


def test_csv_reporter_save_keyframe(
    temp_dirs: tuple[str, str, str],
) -> None:
    """save_keyframe must save frame as a readable JPEG file."""
    _, images_dir, _ = temp_dirs
    frame = np.full((100, 150, 3), fill_value=128, dtype=np.uint8)

    filename, full_path = save_keyframe(
        frame=frame,
        output_dir=images_dir,
        timestamp="2026-10-08_12-30-00",
    )

    assert os.path.exists(full_path)
    assert filename.endswith(".jpg")
    loaded = cv2.imread(full_path)
    assert loaded is not None
    assert loaded.shape == (100, 150, 3)


def test_csv_reporter_read_incidents_csv_roundtrip(
    temp_dirs: tuple[str, str, str],
) -> None:
    """read_incidents_csv returns empty frame for missing file, parses existing data."""
    _, _, reports_dir = temp_dirs

    # 1. Missing file returns empty DataFrame with expected columns
    empty_df = read_incidents_csv(reports_dir=reports_dir, csv_filename="non_existent.csv")
    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty
    expected_cols = [
        "incident_id",
        "start_timestamp",
        "end_timestamp",
        "class",
        "max_confidence",
        "image_path",
    ]
    assert list(empty_df.columns) == expected_cols

    # 2. Write incidents and read back
    csv_file = os.path.join(reports_dir, "incidents.csv")
    data = [
        {
            "incident_id": "inc_001",
            "start_timestamp": "0.5s",
            "end_timestamp": "3.5s",
            "class": "car_car_accident",
            "max_confidence": 0.95,
            "image_path": "/path/to/keyframe_1.jpg",
        },
        {
            "incident_id": "inc_002",
            "start_timestamp": "12.0s",
            "end_timestamp": "15.0s",
            "class": "bike_object_accident",
            "max_confidence": 0.88,
            "image_path": "/path/to/keyframe_2.jpg",
        },
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    loaded_df = read_incidents_csv(reports_dir=reports_dir, csv_filename="incidents.csv")
    assert len(loaded_df) == 2
    assert loaded_df["incident_id"].tolist() == ["inc_001", "inc_002"]
    assert loaded_df["class"].tolist() == ["car_car_accident", "bike_object_accident"]


def test_csv_reporter_list_reports_discovery(
    temp_dirs: tuple[str, str, str],
) -> None:
    """list_reports must discover persisted CSVs and assemble ReportRecord instances."""
    _, images_dir, reports_dir = temp_dirs

    save_report_csv(
        class_name="car_person_accident",
        image_paths=[],
        description="Pedestrian collision",
        confidence=0.91,
        reports_dir=reports_dir,
        timestamp="2026-10-08_11-00-00",
    )
    save_report_csv(
        class_name="car_car_accident",
        image_paths=[],
        description="Two-vehicle collision",
        confidence=0.85,
        reports_dir=reports_dir,
        timestamp="2026-10-08_11-05-00",
    )

    records = list_reports(reports_dir=reports_dir, report_images_dir=images_dir)
    assert len(records) == 2
    assert records[0].class_name in ("car_person_accident", "car_car_accident")
    assert records[0].confidence is not None


# -----------------------------------------------------------------------------
# 4. LLM.py Fallback when API Key is missing (Mock Network)
# -----------------------------------------------------------------------------
def test_llm_fallback_when_api_key_missing_network_strictly_mocked(
    dummy_image_file: str,
) -> None:
    """When API key is missing, return valid fallback without any network calls."""

    def forbidden_network_call(*args, **kwargs):
        raise AssertionError("Network call attempted when API key was missing!")

    # Strictly mock all standard network entrypoints
    with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False), \
         patch.object(socket.socket, "connect", side_effect=forbidden_network_call), \
         patch.object(requests.Session, "send", side_effect=forbidden_network_call), \
         patch.object(urllib.request, "urlopen", side_effect=forbidden_network_call), \
         patch("google.genai.Client", side_effect=forbidden_network_call):

        report = generate_dispatch_report(
            image_path=dummy_image_file,
            detections="car_car_accident",
            timestamp="2026-10-08 14:00:00 UTC",
            lang="en",
            api_key="",
        )

    # Must return a valid structured dict
    assert isinstance(report, dict)

    # Validate against Pydantic schema
    validated = DispatchReportModel(**report)
    assert validated.severity in ("low", "medium", "high")
    assert len(validated.vehicles) > 0
    assert len(validated.recommended_units) > 0
    for u in validated.recommended_units:
        assert u in ("EMS", "police", "fire", "tow")

    # Rendered text check
    assert "rendered_text" in report
    rendered = report["rendered_text"]
    assert "EMERGENCY DISPATCH REPORT" in rendered
    lines = [line.strip() for line in rendered.strip().split("\n") if line.strip()]
    assert 5 <= len(lines) <= 8


def test_llm_fallback_arabic_when_api_key_missing(
    dummy_image_file: str,
) -> None:
    """Arabic fallback report when API key is missing contains valid Arabic dispatch text."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False), \
         patch("google.genai.Client", side_effect=RuntimeError("Network blocked")):

        report = generate_dispatch_report(
            image_path=dummy_image_file,
            detections="car_person_accident",
            timestamp="2026-10-08 14:00:00 UTC",
            lang="ar",
            api_key="",
        )

    assert isinstance(report, dict)
    assert report["severity"] == "high"
    assert "EMS" in report["recommended_units"]
    assert "rendered_text" in report
    rendered = report["rendered_text"]
    assert "تقرير إرسال طوارئ" in rendered
    lines = [line.strip() for line in rendered.strip().split("\n") if line.strip()]
    assert 5 <= len(lines) <= 8


# -----------------------------------------------------------------------------
# 5. i18n Key Parity Tests between English and Arabic
# -----------------------------------------------------------------------------
def test_i18n_ui_strings_key_parity() -> None:
    """UI_STRINGS['en'] and UI_STRINGS['ar'] must have identical key sets."""
    en_keys = set(UI_STRINGS["en"].keys())
    ar_keys = set(UI_STRINGS["ar"].keys())

    missing_in_ar = en_keys - ar_keys
    missing_in_en = ar_keys - en_keys
    diff = en_keys ^ ar_keys

    assert diff == set(), (
        f"i18n key mismatch between English and Arabic!\n"
        f"Missing in Arabic: {sorted(list(missing_in_ar))}\n"
        f"Missing in English: {sorted(list(missing_in_en))}"
    )


def test_i18n_ui_strings_non_empty_strings() -> None:
    """All translation entries must be non-empty strings in both catalogs."""
    for lang in ("en", "ar"):
        for key, val in UI_STRINGS[lang].items():
            assert isinstance(val, str), f"{lang}.{key} is not a str"
            assert val.strip() != "", f"{lang}.{key} is empty"


def test_i18n_format_placeholder_parity() -> None:
    """Format string placeholders (e.g., {device}, {results}) must match between en and ar."""
    pattern = re.compile(r"\{([a-zA-Z0-9_]+(?::[^}]+)?)\}")

    for key in UI_STRINGS["en"]:
        en_placeholders = set(pattern.findall(UI_STRINGS["en"][key]))
        ar_placeholders = set(pattern.findall(UI_STRINGS["ar"][key]))
        assert en_placeholders == ar_placeholders, (
            f"Placeholder mismatch for '{key}':\n"
            f"en has: {en_placeholders}\n"
            f"ar has: {ar_placeholders}"
        )


def test_i18n_class_translations_coverage() -> None:
    """All designated CLASS_NAMES and ACCIDENT_CLASSES must have translations."""
    for cls_name in CLASS_NAMES:
        assert cls_name in CLASS_TRANSLATIONS or any(part in CLASS_TRANSLATIONS for part in cls_name.split("_")), (
            f"Class name '{cls_name}' missing from CLASS_TRANSLATIONS"
        )
        translated_ar = translate_class_name(cls_name, lang="ar")
        assert translated_ar != "", f"Empty Arabic translation for {cls_name}"
        assert translated_ar != cls_name, f"Arabic translation returned raw english for {cls_name}"

        # In English mode, translate_class_name returns raw class name
        assert translate_class_name(cls_name, lang="en") == cls_name


def test_i18n_t_function_behavior() -> None:
    """t() lookup helper returns localized string, formats kwargs, and falls back to key."""
    # Standard lookup
    assert t("app_title", lang="en") == UI_STRINGS["en"]["app_title"]
    assert t("app_title", lang="ar") == UI_STRINGS["ar"]["app_title"]

    # Keyword formatting
    formatted_en = t("device_badge", lang="en", device="cuda:0")
    assert formatted_en == "Compute Device: cuda:0"

    formatted_ar = t("device_badge", lang="ar", device="cuda:0")
    assert "cuda:0" in formatted_ar

    # Missing key fallback
    assert t("non_existent_key_xyz", lang="en") == "non_existent_key_xyz"
    assert t("non_existent_key_xyz", lang="ar") == "non_existent_key_xyz"
