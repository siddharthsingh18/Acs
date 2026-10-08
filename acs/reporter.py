"""
acs.reporter
Handles saving accident keyframe images, managing CSV accident reports,
and reading historical reports without any Streamlit dependencies.
"""
from dataclasses import dataclass, field
import glob
import logging
import os
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from acs.config import config

logger = logging.getLogger("acs.reporter")


@dataclass
class ReportRecord:
    """Represents a persisted accident report record."""

    report_filename: str
    report_path: str
    class_name: str
    timestamp: str
    description: Optional[str] = None
    confidence: Optional[float] = None
    image_filenames: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)


def save_keyframe(
    frame: np.ndarray,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Tuple[str, str]:
    """Saves a single video frame as a JPEG image in the report images directory.

    Args:
        frame: BGR image frame as a numpy array.
        output_dir: Directory where the image should be saved.
        timestamp: Optional formatted timestamp string.

    Returns:
        Tuple[str, str]: (image_filename, full_image_path)
    """
    target_dir = output_dir or config.report_images_dir
    os.makedirs(target_dir, exist_ok=True)

    ts = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"image_{ts}.jpg"
    image_path = os.path.join(target_dir, image_filename)

    # Avoid name collision if multiple frames saved in the same second
    counter = 1
    while os.path.exists(image_path):
        image_filename = f"image_{ts}_{counter}.jpg"
        image_path = os.path.join(target_dir, image_filename)
        counter += 1

    cv2.imwrite(image_path, frame)
    return image_filename, image_path


def save_report_csv(
    class_name: str,
    image_paths: List[str],
    description: Optional[str] = None,
    confidence: Optional[float] = None,
    reports_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Tuple[str, str]:
    """Appends an accident detection entry to a CSV report file.

    Args:
        class_name: Detected accident class or joined list of classes.
        image_paths: List of file paths to saved accident keyframe images.
        description: Optional Gemini generated or user-provided summary.
        confidence: Optional model prediction confidence score.
        reports_dir: Directory where the CSV report will be saved.
        timestamp: Optional formatted timestamp string.

    Returns:
        Tuple[str, str]: (report_filename, full_report_path)
    """
    target_dir = reports_dir or config.reports_dir
    os.makedirs(target_dir, exist_ok=True)

    ts = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{ts}.csv"
    report_path = os.path.join(target_dir, report_filename)

    counter = 1
    while os.path.exists(report_path):
        report_filename = f"report_{ts}_{counter}.csv"
        report_path = os.path.join(target_dir, report_filename)
        counter += 1

    report_data = {
        "Class": class_name,
        "Time": ts,
        "Description": description,
        "Images": ";".join(image_paths) if image_paths else "",
    }
    if confidence is not None:
        report_data["Confidence"] = confidence

    report_df = pd.DataFrame([report_data])
    report_df.to_csv(report_path, index=False)

    return report_filename, report_path


def update_report_description(report_path: str, description: str) -> bool:
    """Updates the description column of an existing accident report CSV.

    Args:
        report_path: Full file path to the report CSV.
        description: Text description to record.

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    if not os.path.exists(report_path):
        return False

    try:
        df = pd.read_csv(report_path)
        df["Description"] = description
        df.to_csv(report_path, index=False)
        return True
    except Exception:
        return False


def read_report_csv(report_path: str) -> pd.DataFrame:
    """Reads a report CSV file into a pandas DataFrame.

    Args:
        report_path: Path to the CSV file.

    Returns:
        pd.DataFrame: Parsed contents of the report.

    Raises:
        FileNotFoundError: If the report file does not exist.
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report CSV not found: '{report_path}'")
    return pd.read_csv(report_path)


def list_reports(
    reports_dir: Optional[str] = None,
    report_images_dir: Optional[str] = None,
) -> List[ReportRecord]:
    """Discovers all persisted accident reports and returns structured records.

    Args:
        reports_dir: Directory containing report CSV files.
        report_images_dir: Directory containing associated image files.

    Returns:
        List[ReportRecord]: List of report records sorted newest first.
    """
    target_reports = reports_dir or config.reports_dir
    target_images = report_images_dir or config.report_images_dir

    if not os.path.exists(target_reports):
        return []

    csv_paths = glob.glob(os.path.join(target_reports, "report_*.csv"))
    records: List[ReportRecord] = []

    for path in sorted(csv_paths, reverse=True):
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue

            class_val = str(df["Class"].iloc[0]) if "Class" in df.columns else "Unknown"
            time_val = str(df["Time"].iloc[0]) if "Time" in df.columns else ""
            desc_val = (
                str(df["Description"].iloc[0])
                if "Description" in df.columns and pd.notnull(df["Description"].iloc[0])
                else None
            )
            conf_val = (
                float(df["Confidence"].iloc[0])
                if "Confidence" in df.columns and pd.notnull(df["Confidence"].iloc[0])
                else None
            )

            # Discover associated images from CSV or fallback to images matching timestamp
            image_paths: List[str] = []
            if "Images" in df.columns and pd.notnull(df["Images"].iloc[0]) and df["Images"].iloc[0]:
                for img in str(df["Images"].iloc[0]).split(";"):
                    if img and os.path.exists(img):
                        image_paths.append(img)

            if not image_paths and time_val and os.path.exists(target_images):
                pattern = os.path.join(target_images, f"image_{time_val}*.jpg")
                image_paths = sorted(glob.glob(pattern))

            image_filenames = [os.path.basename(p) for p in image_paths]
            filename = os.path.basename(path)

            records.append(
                ReportRecord(
                    report_filename=filename,
                    report_path=path,
                    class_name=class_val,
                    timestamp=time_val,
                    description=desc_val,
                    confidence=conf_val,
                    image_filenames=image_filenames,
                    image_paths=image_paths,
                )
            )
        except Exception:
            continue

    return records


def read_incidents_csv(
    reports_dir: Optional[str] = None,
    csv_filename: str = "incidents.csv",
) -> pd.DataFrame:
    """Reads the master incidents CSV log into a pandas DataFrame.

    Args:
        reports_dir: Path to directory holding CSV reports.
        csv_filename: Filename of master log.

    Returns:
        pd.DataFrame: Incident records or empty DataFrame if not found.
    """
    target_dir = reports_dir or config.reports_dir
    csv_path = os.path.join(target_dir, csv_filename)
    columns = [
        "incident_id",
        "start_timestamp",
        "end_timestamp",
        "class",
        "max_confidence",
        "image_path",
    ]
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


def generate_report_pdf(
    incident_id: str,
    class_name: str,
    start_timestamp: str,
    end_timestamp: str,
    max_confidence: float,
    image_path: Optional[str] = None,
    dispatch_text: Optional[str] = None,
    lang: str = "en",
) -> bytes:
    """Generates an official emergency accident incident report as PDF bytes.

    Args:
        incident_id: Unique incident identifier.
        class_name: Detected collision class label.
        start_timestamp: Incident start time.
        end_timestamp: Incident end / timeout time.
        max_confidence: Peak model prediction confidence (0.0 to 1.0).
        image_path: Optional path to peak keyframe image.
        dispatch_text: Optional multimodal dispatch text from Gemini.
        lang: Target language ('en' or 'ar').

    Returns:
        bytes: Raw PDF document bytes suitable for download or storage.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_name = "helvetica"

    # Header Banner
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 10, "ACS - CCTV ACCIDENT INCIDENT REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font(font_name, "", 9)
    pdf.cell(0, 5, "Automated Municipal Traffic Surveillance & Emergency Dispatch System", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.line(10, 27, 200, 27)
    pdf.ln(8)

    # Incident Summary Metadata Table
    pdf.set_font(font_name, "B", 12)
    pdf.cell(0, 8, "Incident Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)

    rows = [
        ("Incident ID:", incident_id),
        ("Collision Class:", class_name),
        ("Max Confidence:", f"{max_confidence * 100:.1f}%" if max_confidence <= 1.0 else f"{max_confidence:.1f}%"),
        ("Start Timestamp:", start_timestamp),
        ("End Timestamp:", end_timestamp),
        ("Generated Time:", time.strftime("%Y-%m-%d %H:%M:%S UTC")),
    ]

    for label, val in rows:
        pdf.set_font(font_name, "B", 10)
        pdf.cell(45, 6, label, border=0)
        pdf.set_font(font_name, "", 10)
        # Clean text to safe latin-1 if using helvetica
        safe_val = str(val).encode("latin-1", "replace").decode("latin-1") if font_name == "helvetica" else str(val)
        pdf.cell(0, 6, safe_val, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # Embedded Keyframe Image
    if image_path and os.path.exists(image_path):
        try:
            pdf.set_font(font_name, "B", 12)
            pdf.cell(0, 8, "Peak Collision Keyframe", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            # Embed image scaled nicely
            pdf.image(image_path, w=140)
            pdf.ln(6)
        except Exception:
            pdf.set_font(font_name, "I", 9)
            pdf.cell(0, 6, f"[Keyframe image available at: {image_path}]", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # Multimodal Gemini Emergency Dispatch Assessment
    if dispatch_text:
        pdf.set_font(font_name, "B", 12)
        pdf.cell(0, 8, "Multimodal Emergency Dispatch Assessment (AI)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 9)
        clean_text = (
            dispatch_text.encode("latin-1", "replace").decode("latin-1")
            if font_name == "helvetica"
            else dispatch_text
        )
        pdf.multi_cell(0, 5, clean_text)
        pdf.ln(4)

    # Footer notice
    pdf.set_y(-20)
    pdf.set_font(font_name, "I", 8)
    pdf.cell(0, 8, "Confidential - For Emergency Response and Municipal Traffic Dispatch Use Only", align="C")

    return bytes(pdf.output())
