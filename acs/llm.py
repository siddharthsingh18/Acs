"""
acs.llm
Provides Google Gemini client integration, structured dispatch report generation,
multi-frame context handling, exponential backoff retries, and template fallback.
No Streamlit dependencies are included to ensure testability.
"""
import concurrent.futures
import json
import logging
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from acs.config import config

# Ensure environment variables are loaded
load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)


class DispatchReportModel(BaseModel):
    """Structured accident dispatch report conforming to emergency response requirements."""

    vehicles: List[str] = Field(
        description="List of involved vehicles, including types, make/models, and colors if visible."
    )
    collision_point: str = Field(
        description="Impact location on roadway or intersection, point of collision, and impact orientation."
    )
    damage: str = Field(
        description="Visible vehicular and property damage, crumpled zones, or debris."
    )
    road_conditions: str = Field(
        description="Road surface conditions (dry, wet, icy, debris) and visibility/weather conditions."
    )
    hazards: str = Field(
        description="Active hazards (fuel/fluid leaks, fire, downed lines, blocked traffic lanes)."
    )
    recommended_units: List[Literal["EMS", "police", "fire", "tow"]] = Field(
        description="List of required emergency responder units strictly chosen from: EMS, police, fire, tow."
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Overall incident severity classification: low, medium, or high."
    )


class VerificationReportModel(BaseModel):
    """Structured response model for VLM collision verification."""

    is_accident: bool = Field(
        description="True if a real traffic collision is visible in the provided frames, False otherwise."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 that a real traffic collision occurred.",
    )
    reason: str = Field(
        description="Brief factual explanation of visual evidence or rationale."
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="Incident severity classification: low, medium, or high."
    )
    vehicles_involved: int = Field(
        ge=0,
        description="Count of vehicles visibly involved in the collision."
    )


def extract_class_names(detections: Any) -> List[str]:
    """Normalizes various detection input formats into a list of clean class strings.

    Args:
        detections: Raw detection representations (string, list, DetectionResult, or None).

    Returns:
        List[str]: Cleaned list of class name strings.
    """
    if not detections:
        return []
    if isinstance(detections, str):
        cleaned = detections.replace(";", ",").replace("&", ",")
        return [c.strip().lower() for c in cleaned.split(",") if c.strip()]
    if isinstance(detections, (list, tuple, set)):
        result: List[str] = []
        for item in detections:
            if isinstance(item, str):
                result.append(item.strip().lower())
            elif hasattr(item, "class_name"):
                result.append(str(item.class_name).strip().lower())
        return result
    if hasattr(detections, "detected_classes") and detections.detected_classes:
        return [str(c).strip().lower() for c in detections.detected_classes]
    if hasattr(detections, "accident_classes") and detections.accident_classes:
        return [str(c).strip().lower() for c in detections.accident_classes]
    if hasattr(detections, "detections"):
        return [
            str(box.class_name).strip().lower()
            for box in detections.detections
            if hasattr(box, "class_name")
        ]
    return []


def find_surrounding_frames(
    peak_image_path: str,
    max_before: int = 2,
    max_after: int = 2,
) -> List[str]:
    """Discovers 1-2 frames chronologically before and after the peak keyframe if available.

    Args:
        peak_image_path: Absolute or relative path to the peak keyframe image.
        max_before: Maximum number of prior frames to include.
        max_after: Maximum number of posterior frames to include.

    Returns:
        List[str]: Ordered sequence of existing image paths with peak keyframe centered.
    """
    if not os.path.exists(peak_image_path):
        return [peak_image_path]

    image_dir = os.path.dirname(os.path.abspath(peak_image_path))
    if not os.path.exists(image_dir):
        return [peak_image_path]

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    candidates.sort()

    abs_peak = os.path.abspath(peak_image_path)
    if abs_peak in candidates:
        idx = candidates.index(abs_peak)
        start_idx = max(0, idx - max_before)
        end_idx = min(len(candidates), idx + max_after + 1)
        return candidates[start_idx:end_idx]

    return [peak_image_path]


def render_dispatch_text(
    report_data: Dict[str, Any],
    lang: str = "en",
) -> str:
    """Renders structured dispatch report JSON into a 5-8 line concise emergency dispatch text.

    Args:
        report_data: Dictionary conforming to DispatchReportModel schema.
        lang: Target language ('en' for English, 'ar' for Arabic).

    Returns:
        str: 5-8 line formatted dispatch text.
    """
    vehicles = report_data.get("vehicles", [])
    if isinstance(vehicles, list):
        vehicles_en = ", ".join(vehicles) if vehicles else "Unspecified vehicles"
    else:
        vehicles_en = str(vehicles)

    collision_point = str(report_data.get("collision_point", "Monitored roadway zone"))
    damage = str(report_data.get("damage", "Impact damage observed"))
    road_conditions = str(report_data.get("road_conditions", "Standard roadway conditions"))
    hazards = str(report_data.get("hazards", "Traffic obstruction"))
    units = report_data.get("recommended_units", ["police", "tow"])
    units_str = ", ".join(units).upper() if isinstance(units, list) else str(units).upper()
    severity = str(report_data.get("severity", "medium")).upper()
    ts = str(report_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC")))

    if lang == "ar":
        sev_map = {"HIGH": "عالية", "MEDIUM": "متوسطة", "LOW": "منخفضة"}
        sev_ar = sev_map.get(severity, severity)

        unit_map = {
            "EMS": "إسعاف (EMS)",
            "POLICE": "شرطة (Police)",
            "FIRE": "دفاع مدني / إطفاء (Fire)",
            "TOW": "سحب مركبات (Tow)",
        }
        if isinstance(units, list):
            units_ar = ", ".join([unit_map.get(u.upper(), u) for u in units])
        else:
            units_ar = str(units)

        lines = [
            "[تقرير إرسال طوارئ - حادث مروري]",
            f"الوقت: {ts} | مستوى الخطورة: {sev_ar}",
            f"المركبات المعنية: {vehicles_en}",
            f"نقطة الاصطدام: {collision_point}",
            f"الأضرار والمخاطر: {damage} | {hazards}",
            f"حالة الطريق: {road_conditions}",
            f"الوحدات الموصى بإرسالها: {units_ar}",
        ]
        return "\n".join(lines)

    lines = [
        "[EMERGENCY DISPATCH REPORT - TRAFFIC COLLISION]",
        f"Timestamp: {ts} | Severity: {severity}",
        f"Vehicles Involved: {vehicles_en}",
        f"Collision Point: {collision_point}",
        f"Observed Damage: {damage}",
        f"Hazards & Roadway: {hazards} | {road_conditions}",
        f"Recommended Units: {units_str}",
    ]
    return "\n".join(lines)


def build_fallback_report(
    detections: Optional[Any] = None,
    timestamp: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """Builds a structured fallback dispatch report directly from YOLO detections.

    Used when the Gemini API is unreachable, times out, or when credentials are missing.

    Args:
        detections: Raw detection outputs or class labels.
        timestamp: Incident timestamp.
        lang: Target output language.

    Returns:
        Dict[str, Any]: Structured dispatch report dictionary conforming to DispatchReportModel.
    """
    class_names = extract_class_names(detections)
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC")

    has_person = any("person" in c for c in class_names)
    has_bike = any("bike" in c for c in class_names)
    has_car = any("car" in c for c in class_names)
    has_injury = any("injury" in c for c in class_names)

    vehicles: List[str] = []
    if "car_car_accident" in class_names:
        vehicles = ["passenger car 1", "passenger car 2"]
    elif "car_bike_accident" in class_names:
        vehicles = ["passenger car", "motorcycle / bicycle"]
    elif "bike_bike_accident" in class_names:
        vehicles = ["motorcycle / bicycle 1", "motorcycle / bicycle 2"]
    elif "car_person_accident" in class_names:
        vehicles = ["passenger car", "pedestrian involved"]
    elif "bike_person_accident" in class_names:
        vehicles = ["motorcycle / bicycle", "pedestrian involved"]
    elif "car_object_accident" in class_names:
        vehicles = ["passenger car (roadside object impact)"]
    else:
        if has_car and has_bike:
            vehicles = ["passenger car", "motorcycle / bicycle"]
        elif has_car:
            vehicles = ["passenger car"]
        elif has_bike:
            vehicles = ["motorcycle / bicycle"]
        else:
            vehicles = ["unidentified motor vehicle"]

    collision_point = "Active roadway intersection / traffic camera surveillance zone"
    detected_summary = ", ".join(class_names) if class_names else "traffic collision"
    damage = f"Impact damage detected via vision model matching: {detected_summary}"
    road_conditions = "Standard asphalt roadway conditions observed in camera feed"

    if has_person or has_injury:
        hazards = "Pedestrian / vulnerable road user impacted on active roadway; high collision risk"
    elif has_bike:
        hazards = "Downed two-wheeler rider and debris obstructing active lane"
    else:
        hazards = "Disabled collision vehicles obstructing active roadway lane"

    recommended_units: List[str] = []
    if has_person or has_injury or "car_person_accident" in class_names or "bike_person_accident" in class_names:
        recommended_units = ["EMS", "police", "tow"]
    elif has_bike:
        recommended_units = ["EMS", "police", "tow"]
    else:
        recommended_units = ["police", "tow"]

    if has_person or has_injury or "car_person_accident" in class_names or "bike_person_accident" in class_names:
        severity = "high"
    elif (
        "car_car_accident" in class_names
        or "car_bike_accident" in class_names
        or "bike_bike_accident" in class_names
        or "car_object_accident" in class_names
    ):
        severity = "medium"
    else:
        severity = "low"

    report_dict: Dict[str, Any] = {
        "vehicles": vehicles,
        "collision_point": collision_point,
        "damage": damage,
        "road_conditions": road_conditions,
        "hazards": hazards,
        "recommended_units": recommended_units,
        "severity": severity,
        "timestamp": ts,
    }

    report_dict["rendered_text"] = render_dispatch_text(report_dict, lang=lang)
    return report_dict


def build_dispatch_prompt(
    num_frames: int,
    detections: Optional[Any] = None,
    timestamp: Optional[str] = None,
    lang: str = "en",
) -> str:
    """Constructs prompt for Gemini multi-frame analysis and structured dispatch output.

    Args:
        num_frames: Number of sequential images provided to the model.
        detections: Detections from YOLO.
        timestamp: Timestamp of the event.
        lang: Target language.

    Returns:
        str: Prompt text.
    """
    class_names = extract_class_names(detections)
    detected_str = ", ".join(class_names) if class_names else "Automated traffic collision detection"
    ts_str = timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC")

    frame_desc = (
        f"{num_frames} sequential camera frames (pre-collision approach, peak impact keyframe, and post-collision rest)"
        if num_frames > 1
        else "1 peak keyframe image of the traffic collision"
    )

    prompt = (
        f"You are an expert traffic incident and 911 dispatch multimodal analysis system.\n"
        f"You are provided {frame_desc}.\n"
        f"Incident timestamp: {ts_str}.\n"
        f"Computer vision detections: {detected_str}.\n\n"
        f"Analyze all provided frames carefully and extract the structured dispatch fields:\n"
        f"1. vehicles: List all involved vehicles, noting body types, colors, and lane positioning.\n"
        f"2. collision_point: Precise location of collision impact on roadway/intersection and angle.\n"
        f"3. damage: Visible structural damage, vehicle deformity, intrusion, or broken components.\n"
        f"4. road_conditions: Road surface state (dry, wet, icy, debris, markings) and visibility.\n"
        f"5. hazards: Immediate roadside dangers (leaking fluids, fire, entrapment, blocked lanes).\n"
        f"6. recommended_units: Required emergency responder units strictly chosen from ['EMS', 'police', 'fire', 'tow'].\n"
        f"7. severity: Overall incident severity strictly chosen from 'low', 'medium', 'high'.\n"
    )
    if lang == "ar":
        prompt += "\nProvide textual descriptions inside the JSON fields in Arabic."
    return prompt


def _call_gemini_api(
    client: Any,
    model_name: str,
    contents: List[Any],
) -> str:
    """Executes the Google Gemini generate_content call using structured JSON schema."""
    config_obj = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DispatchReportModel,
        temperature=0.2,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config_obj,
    )
    return response.text or ""


def generate_dispatch_report(
    image_path: Union[str, List[str]],
    detections: Optional[Any] = None,
    timestamp: Optional[str] = None,
    lang: str = "en",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    additional_image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generates structured emergency dispatch report using Google Gemini multimodal AI.

    Features:
    - Structured output JSON mode adhering to DispatchReportModel schema.
    - Multi-frame input: peak keyframe + 1-2 frames before/after if available.
    - Automatic exponential backoff retries.
    - Hard timeout enforcement per attempt.
    - Graceful fallback to template report from YOLO detections when API is unavailable.
    - Rendered 5-8 line dispatch text in English or Arabic.

    Args:
        image_path: Peak keyframe file path, or list of sequential frame paths.
        detections: YOLO detection classes or DetectionResult.
        timestamp: Incident timestamp string.
        lang: Language for rendered report ('en' or 'ar').
        api_key: Optional Gemini API key override (defaults to GOOGLE_API_KEY env).
        model_name: Gemini model name override (defaults to config.gemini_model_name).
        timeout: Maximum seconds to wait per call before timing out.
        max_retries: Number of retry attempts on failure.
        additional_image_paths: Optional extra surrounding frame paths.

    Returns:
        Dict[str, Any]: Structured dispatch report dictionary with schema fields and rendered_text.
    """
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC")
    active_key = api_key or os.getenv("GOOGLE_API_KEY")

    # Fast graceful fallback if key is missing or SDK is unavailable
    if not active_key or genai is None or types is None:
        logger.info("Google GenAI key or SDK unavailable; using fallback template report.")
        return build_fallback_report(detections=detections, timestamp=ts, lang=lang)

    # Resolve frame paths: peak + 1-2 before/after if available
    frame_paths: List[str] = []
    if isinstance(image_path, list):
        frame_paths.extend(image_path)
    else:
        frame_paths.extend(find_surrounding_frames(image_path, max_before=2, max_after=2))

    if additional_image_paths:
        frame_paths.extend(additional_image_paths)

    # Filter to existing readable files and deduplicate preserving order
    seen: set = set()
    valid_frames: List[str] = []
    for fp in frame_paths:
        if fp and os.path.exists(fp) and fp not in seen:
            seen.add(fp)
            valid_frames.append(fp)

    if not valid_frames:
        logger.warning("No valid image frames found on disk; using fallback template report.")
        return build_fallback_report(detections=detections, timestamp=ts, lang=lang)

    # Limit to maximum 5 frames (peak + 2 before + 2 after)
    selected_frames = valid_frames[:5]

    # Prepare inline image parts
    image_parts: List[Any] = []
    for fp in selected_frames:
        try:
            with open(fp, "rb") as f:
                img_data = f.read()
            mime_type = mimetypes.guess_type(fp)[0] or "image/jpeg"
            part = types.Part.from_bytes(data=img_data, mime_type=mime_type)
            image_parts.append(part)
        except Exception as read_err:
            logger.warning(f"Failed to read image frame '{fp}': {read_err}")

    if not image_parts:
        return build_fallback_report(detections=detections, timestamp=ts, lang=lang)

    chosen_model = model_name or config.gemini_model_name
    call_timeout = (
        timeout
        if timeout is not None
        else getattr(config, "gemini_timeout_seconds", 15.0)
    )
    retries = (
        max_retries
        if max_retries is not None
        else getattr(config, "gemini_max_retries", 3)
    )

    prompt = build_dispatch_prompt(
        num_frames=len(image_parts),
        detections=detections,
        timestamp=ts,
        lang=lang,
    )
    contents = image_parts + [prompt]

    # Initialize Gemini client
    client = genai.Client(api_key=active_key)

    # Execute with exponential backoff retries and timeout
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _call_gemini_api,
                    client=client,
                    model_name=chosen_model,
                    contents=contents,
                )
                raw_response = future.result(timeout=call_timeout)

            # Strip possible markdown wrapper if returned
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE
            )
            parsed_data = json.loads(cleaned)

            # Validate and fill missing fields against schema defaults
            report_dict: Dict[str, Any] = {
                "vehicles": parsed_data.get("vehicles", ["unspecified vehicle"]),
                "collision_point": parsed_data.get("collision_point", "Monitored roadway"),
                "damage": parsed_data.get("damage", "Impact damage observed"),
                "road_conditions": parsed_data.get("road_conditions", "Standard roadway"),
                "hazards": parsed_data.get("hazards", "Traffic obstruction"),
                "recommended_units": parsed_data.get("recommended_units", ["police", "tow"]),
                "severity": parsed_data.get("severity", "medium"),
                "timestamp": ts,
            }
            report_dict["rendered_text"] = render_dispatch_text(report_dict, lang=lang)
            return report_dict

        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Gemini API attempt {attempt + 1}/{retries + 1} failed: {exc}"
            )
            if attempt < retries:
                backoff_delay = 1.0 * (2 ** attempt)
                time.sleep(backoff_delay)

    logger.error(
        f"All {retries + 1} Gemini API attempts failed ({last_error}). Triggering graceful fallback."
    )
    return build_fallback_report(detections=detections, timestamp=ts, lang=lang)


def generate_accident_report(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    yolo_report: Optional[str] = None,
    incident_time: Optional[str] = None,
    lang: str = "en",
) -> str:
    """Backwards-compatible wrapper calling generate_dispatch_report and returning rendered text.

    Args:
        image_path: Path to the keyframe image on disk.
        api_key: Optional Google API key.
        model_name: Gemini model name.
        yolo_report: Detected YOLO class names.
        incident_time: Timestamp of detection.
        lang: Language for the generated report ('en' or 'ar').

    Returns:
        str: 5-8 line rendered dispatch text.
    """
    report = generate_dispatch_report(
        image_path=image_path,
        detections=yolo_report,
        timestamp=incident_time,
        lang=lang,
        api_key=api_key,
        model_name=model_name,
    )
    return report.get("rendered_text") or render_dispatch_text(report, lang=lang)


def verify_incident_frames_vlm(
    frames: List[Union[str, np.ndarray, bytes]],
    detections: Optional[Any] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    """Sends 3 to 5 chronological frames to Google Gemini VLM for collision verification.

    Prompt format:
    "Is a real traffic collision visible? Answer JSON: {is_accident: bool, confidence: 0-1, reason: str, severity: low|medium|high, vehicles_involved: int}"

    Args:
        frames: 3 to 5 images (file paths, numpy BGR frames, or JPEG bytes) representing:
            before impact, at peak collision, and after impact.
        detections: Optional YOLO detection classes or DetectionResult for fallback context.
        api_key: Optional Gemini API key override (defaults to GOOGLE_API_KEY or GEMINI_API_KEY).
        model_name: Model override (defaults to config.gemini_model_name).
        timeout: Maximum seconds per API call attempt before timing out.
        max_retries: Number of exponential backoff retry attempts.

    Returns:
        Dict[str, Any]: Structured dictionary conforming to VerificationReportModel schema.
    """
    import cv2

    active_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not active_key or genai is None or types is None:
        logger.info("Google GenAI API key or SDK unavailable for VLM verification; using fallback.")
        return {
            "is_accident": True,
            "confidence": 0.5,
            "reason": "VLM API key or SDK unavailable; graceful fallback to YOLO",
            "severity": "medium",
            "vehicles_involved": 2,
            "verification_status": "unverified",
        }

    image_parts: List[Any] = []
    for f in (frames or [])[:5]:
        try:
            if isinstance(f, str):
                if os.path.exists(f):
                    with open(f, "rb") as fp:
                        data = fp.read()
                    mime_type = mimetypes.guess_type(f)[0] or "image/jpeg"
                    image_parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            elif isinstance(f, np.ndarray):
                ret, buf = cv2.imencode(".jpg", f)
                if ret:
                    image_parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
            elif isinstance(f, bytes):
                image_parts.append(types.Part.from_bytes(data=f, mime_type="image/jpeg"))
        except Exception as err:
            logger.warning(f"Failed to prepare frame for VLM verification: {err}")

    if not image_parts:
        logger.warning("No valid frames available for VLM verification; using fallback.")
        return {
            "is_accident": True,
            "confidence": 0.5,
            "reason": "No valid frames available for VLM verification; fallback",
            "severity": "medium",
            "vehicles_involved": 2,
            "verification_status": "unverified",
        }

    strict_prompt = (
        "You are an expert traffic incident verification multimodal system.\n"
        f"You are provided {len(image_parts)} chronological camera frames from a potential incident: "
        "before impact, at peak collision, and after impact.\n\n"
        "Carefully examine vehicle kinematics, road positioning, structural deformities, and traffic dynamics.\n"
        "Question: Is a real traffic collision visible?\n"
        'Answer JSON: {"is_accident": bool, "confidence": 0-1, "reason": str, "severity": low|medium|high, "vehicles_involved": int}'
    )

    chosen_model = model_name or config.gemini_model_name
    call_timeout = timeout if timeout is not None else getattr(config, "vlm_timeout_seconds", 10.0)
    retries = max_retries if max_retries is not None else getattr(config, "vlm_max_retries", 2)

    try:
        client = genai.Client(api_key=active_key)
    except Exception as init_err:
        logger.warning(f"Failed to initialize GenAI client: {init_err}")
        return {
            "is_accident": True,
            "confidence": 0.5,
            "reason": f"GenAI client initialization failed: {init_err}",
            "severity": "medium",
            "vehicles_involved": 2,
            "verification_status": "unverified",
        }

    def _call():
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationReportModel,
            temperature=0.1,
        )
        resp = client.models.generate_content(
            model=chosen_model,
            contents=[*image_parts, strict_prompt],
            config=cfg,
        )
        return resp.text or ""

    last_error = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                raw_text = future.result(timeout=call_timeout)
            vlm_duration = time.time() - t0

            # Clean markdown code block fences if present
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

            parsed = json.loads(cleaned_text)
            validated = VerificationReportModel(**parsed)
            try:
                from backend.app.core.metrics import record_vlm_call
                record_vlm_call(status="success", duration_seconds=vlm_duration, model=chosen_model)
            except Exception:
                pass
            return {
                "is_accident": bool(validated.is_accident),
                "confidence": float(validated.confidence),
                "reason": str(validated.reason),
                "severity": str(validated.severity).lower(),
                "vehicles_involved": int(validated.vehicles_involved),
                "verification_status": "verified",
            }
        except concurrent.futures.TimeoutError:
            last_error = TimeoutError(f"Call timed out after {call_timeout}s")
            logger.warning(f"VLM verification attempt {attempt + 1}/{retries + 1} timed out")
            try:
                from backend.app.core.metrics import record_vlm_call
                record_vlm_call(status="timeout", duration_seconds=time.time() - t0, model=chosen_model)
            except Exception:
                pass
        except Exception as exc:
            last_error = exc
            logger.warning(f"VLM verification attempt {attempt + 1}/{retries + 1} failed: {exc}")
            try:
                from backend.app.core.metrics import record_vlm_call
                record_vlm_call(status="failure", duration_seconds=time.time() - t0, model=chosen_model)
            except Exception:
                pass

        if attempt < retries:
            time.sleep(1.0 * (2 ** attempt))

    logger.error(f"All VLM verification attempts failed ({last_error}). Triggering unverified fallback.")
    return {
        "is_accident": True,
        "confidence": 0.5,
        "reason": f"VLM verification failed ({last_error}); falling back to unverified YOLO detection",
        "severity": "medium",
        "vehicles_involved": 2,
        "verification_status": "unverified",
    }

