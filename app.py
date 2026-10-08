"""
app.py
Unified Streamlit entry point for the Automated CCTV Accident Detection System (ACS).
Features:
- Sidebar controls: Language switcher (English / Arabic), dynamic model scanner (.pt),
  confidence slider, and disabled injury model toggle when injury.pt is absent.
- Conditional RTL CSS applied strictly when Arabic is active.
- Pages/Tabs: Live Analysis (with ByteTrack tracking & progress bar),
  Reports (CSV table, keyframe thumbnails, Gemini dispatch assessment, PDF/CSV download),
  and About.
- Cached model loading via @st.cache_resource and state management via st.session_state.
- Complete internationalization via acs.i18n.
"""
import logging
import os
import time
from typing import List, Optional
import cv2
import streamlit as st

from acs.alerts import send_incident_alert, send_test_alert
from acs.config import ACCIDENT_CLASSES, CLASS_NAMES, config
from acs.detector import AccidentDetector, AccidentEventManager, AccidentIncident
from acs.i18n import t, translate_class_name
from acs.llm import generate_dispatch_report, render_dispatch_text
from acs.reporter import generate_report_pdf, read_incidents_csv

logger = logging.getLogger("acs.app")


# -----------------------------------------------------------------------------
# 1. Page Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ACS - CCTV Accident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

config.ensure_directories()


# -----------------------------------------------------------------------------
# 2. Model Scanner & Resource Caching
# -----------------------------------------------------------------------------
def scan_model_weights() -> List[str]:
    """Scans the repository root for available YOLO .pt model weights files.

    Returns:
        List[str]: Sorted list of relative paths to existing .pt files.
    """
    models = []
    for item in sorted(os.listdir(".")):
        if item.endswith(".pt") and os.path.isfile(item) and not item.startswith("."):
            models.append(item)
    return models or ["best.pt"]


@st.cache_resource(show_spinner=False)
def get_cached_detector(
    model_path: str,
    injury_model_path: Optional[str],
    confidence_threshold: float,
    device: str,
) -> AccidentDetector:
    """Loads and caches the AccidentDetector to avoid reloading models on rerun.

    Args:
        model_path: Path to the primary YOLO weights file.
        injury_model_path: Optional path to the injury weights file.
        confidence_threshold: Detection cutoff threshold.
        device: Target hardware device ('cuda:0', 'mps', or 'cpu').

    Returns:
        AccidentDetector: Cached detector instance.
    """
    return AccidentDetector(
        model_path=model_path,
        injury_model_path=injury_model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Internationalization
# -----------------------------------------------------------------------------
def render_sidebar() -> str:
    """Renders the application sidebar and returns the active language code.

    Returns:
        str: Active language identifier ('en' or 'ar').
    """
    st.sidebar.title(t("sidebar_title", st.session_state.get("lang", "en")))

    # Language Switcher
    current_lang = st.session_state.get("lang", "en")
    lang_choice = st.sidebar.radio(
        label=t("lang_select", current_lang),
        options=["English", "العربية"],
        index=0 if current_lang == "en" else 1,
    )
    lang = "en" if lang_choice == "English" else "ar"
    st.session_state["lang"] = lang

    st.sidebar.markdown("---")
    st.sidebar.subheader(t("active_settings_header", lang))

    # Model Selector (dynamically scans for .pt files)
    available_models = scan_model_weights()
    default_idx = available_models.index("best.pt") if "best.pt" in available_models else 0
    selected_model = st.sidebar.selectbox(
        label=t("model_select", lang),
        options=available_models,
        index=default_idx,
    )
    st.session_state["selected_model"] = selected_model

    # Confidence Threshold Slider
    confidence_thresh = st.sidebar.slider(
        label=t("confidence_slider", lang),
        min_value=0.10,
        max_value=0.95,
        value=float(st.session_state.get("confidence_threshold", config.confidence_threshold)),
        step=0.05,
    )
    st.session_state["confidence_threshold"] = confidence_thresh

    # Optional Injury Model Checkbox (disabled with notice if injury.pt is missing)
    injury_pt_path = "injury.pt"
    injury_pt_exists = os.path.exists(injury_pt_path)

    use_injury = st.sidebar.checkbox(
        label=t("use_injury_checkbox", lang),
        value=False if not injury_pt_exists else st.session_state.get("use_injury", False),
        disabled=not injury_pt_exists,
    )
    st.session_state["use_injury"] = use_injury and injury_pt_exists

    if not injury_pt_exists:
        st.sidebar.caption(f"⚠️ {t('injury_disabled_notice', lang)}")

    st.sidebar.markdown("---")
    # Device indicator badge
    st.sidebar.info(t("device_badge", lang, device=config.device.upper()))

    st.sidebar.markdown("---")
    st.sidebar.subheader(t("alerts_section_title", lang))

    enable_alerts = st.sidebar.checkbox(
        label=t("enable_alerts_toggle", lang),
        value=st.session_state.get("enable_alerts", config.alerts_enabled),
    )
    st.session_state["enable_alerts"] = enable_alerts

    dry_run_alerts = st.sidebar.checkbox(
        label=t("dry_run_toggle", lang),
        value=st.session_state.get("dry_run_alerts", config.alerts_dry_run),
    )
    st.session_state["dry_run_alerts"] = dry_run_alerts

    if st.sidebar.button(t("send_test_alert_btn", lang), use_container_width=True):
        with st.sidebar:
            with st.spinner("Dispatching test alert..."):
                test_results = send_test_alert(dry_run=dry_run_alerts, lang=lang)
                status_str = ", ".join([f"{k}: {'OK' if v else 'FAIL'}" for k, v in test_results.items()])
                st.success(t("test_alert_success", lang, results=status_str))

    return lang


# -----------------------------------------------------------------------------
# 4. RTL Layout Handler
# -----------------------------------------------------------------------------
def apply_rtl_css_if_arabic(lang: str) -> None:
    """Injects right-to-left layout CSS strictly when Arabic is selected.

    Args:
        lang: Current active language code.
    """
    if lang == "ar":
        st.markdown(
            """
            <style>
            .stApp, .element-container, .stMarkdown, .stText, .stButton, div[data-testid="stSidebar"],
            div[data-testid="stRadio"], div[data-testid="stSelectbox"], div[data-testid="stSlider"],
            div[data-testid="stCheckbox"], div[data-testid="stFileUploader"], .stTabs,
            div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
                direction: rtl !important;
                text-align: right !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                direction: rtl !important;
                justify-content: flex-start !important;
            }
            div[data-testid="stVideo"] {
                direction: ltr !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 5. Tab 1: Live Analysis View
# -----------------------------------------------------------------------------
def render_live_analysis_tab(detector: AccidentDetector, lang: str) -> None:
    """Renders the Live Analysis tab for video stream processing and tracking.

    Args:
        detector: Loaded and cached AccidentDetector.
        lang: Active language ('en' or 'ar').
    """
    st.header(t("analysis_header", lang))
    st.caption(t("analysis_subtitle", lang))

    # Video Source Selector: Upload or Preloaded Sample
    col_source, col_run = st.columns([3, 1])
    with col_source:
        video_source_mode = st.radio(
            label=t("video_source_label", lang),
            options=[t("upload_option", lang), t("sample_option", lang)],
            horizontal=True,
        )

    target_video_path: Optional[str] = None

    if video_source_mode == t("sample_option", lang):
        sample_path = "acc_ex.mp4"
        if os.path.exists(sample_path):
            target_video_path = sample_path
            st.info(f"Loaded sample video: `{sample_path}`")
        else:
            st.warning("Sample clip `acc_ex.mp4` not found in workspace.")
    else:
        uploaded_file = st.file_uploader(
            label=t("upload_label", lang),
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_file is not None:
            saved_temp_path = os.path.join(config.temp_dir, "input_analysis.mp4")
            with open(saved_temp_path, "wb") as f:
                f.write(uploaded_file.read())
            target_video_path = saved_temp_path

    if not target_video_path:
        return

    with col_run:
        st.write("")
        st.write("")
        run_btn = st.button(t("start_analysis_btn", lang), type="primary", use_container_width=True)

    if not run_btn:
        return

    # Initialize video capture
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        st.error("Failed to open video file for processing.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Layout: Video Frame Column & Live Incident Feed Column
    col_video, col_feed = st.columns([3, 2])

    with col_video:
        st.subheader(t("video_player_col", lang))
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_caption = st.empty()

    with col_feed:
        st.subheader(t("incident_feed_col", lang))
        feed_placeholder = st.empty()

    # Initialize AccidentEventManager
    event_manager = AccidentEventManager(
        consecutive_frames_threshold=5,
        timeout_seconds=3.0,
        cooldown_seconds=config.cooldown_seconds,
        confidence_threshold=detector.confidence_threshold,
        output_dir=config.report_images_dir,
        reports_dir=config.reports_dir,
        auto_save=True,
    )

    detected_incidents: List[AccidentIncident] = []
    frame_idx = 0
    start_wall_time = time.time()

    def update_feed_display():
        with feed_placeholder.container():
            if not detected_incidents:
                st.info(t("no_incidents_yet", lang))
            else:
                for idx, inc in enumerate(reversed(detected_incidents)):
                    card_title = translate_class_name(inc.class_name, lang=lang)
                    with st.expander(f"🚨 {card_title} ({inc.incident_id})", expanded=(idx == 0)):
                        if inc.image_path and os.path.exists(inc.image_path):
                            st.image(inc.image_path, use_container_width=True)
                        st.write(f"**{t('timestamp_label', lang, timestamp=inc.start_timestamp)}**")
                        st.write(f"**{t('confidence_label', lang, confidence=inc.max_confidence)}**")
                        duration = max(0.1, inc.end_time - inc.start_time)
                        st.write(f"**{t('duration_label', lang, duration=duration)}**")

    update_feed_display()

    # Video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1

        # Run ByteTrack tracking inference
        result = detector.track_frame(frame)

        # Update video placeholder with annotated frame
        rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    def handle_incident_closure(incident: AccidentIncident):
        """Processes closed incident: updates feed, generates dispatch report, and sends alerts."""
        detected_incidents.append(incident)
        update_feed_display()

        cls_localized = translate_class_name(incident.class_name, lang=lang)
        st.toast(
            t(
                "incident_detected_toast",
                lang,
                incident_id=incident.incident_id,
                class_name=cls_localized,
            ),
            icon="🚨",
        )

        # 1. Generate multimodal dispatch assessment report
        report_text = ""
        try:
            dispatch_data = generate_dispatch_report(
                image_path=incident.image_path,
                detections=incident.class_name,
                timestamp=incident.start_timestamp,
                lang=lang,
            )
            report_text = dispatch_data.get("rendered_text") or render_dispatch_text(dispatch_data, lang=lang)
        except Exception:
            report_text = (
                f"[ACS DISPATCH ALERT] {cls_localized.upper()}\n"
                f"Incident ID: {incident.incident_id}\n"
                f"Time: {incident.start_timestamp}\n"
                f"Confidence: {incident.max_confidence:.1%}"
            )

        # 2. Dispatch alert with keyframe and report text
        if st.session_state.get("enable_alerts", config.alerts_enabled):
            try:
                alert_res = send_incident_alert(
                    incident=incident,
                    report_text=report_text,
                    image_path=incident.image_path,
                    dry_run=st.session_state.get("dry_run_alerts", config.alerts_dry_run),
                )
                channels_str = ", ".join([f"{k} ({'✓' if v else '✗'})" for k, v in alert_res.items()])
                st.toast(
                    t(
                        "alert_dispatched_toast",
                        lang,
                        incident_id=incident.incident_id,
                        channels=channels_str,
                    ),
                    icon="📨",
                )
            except Exception:
                # Alerts must never crash the pipeline
                pass

    # Video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_idx += 1

        # Run ByteTrack tracking inference
        result = detector.track_frame(frame)

        # Update video placeholder with annotated frame
        rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Update event manager state machine
        completed_incident = event_manager.update(result, frame=frame, timestamp=timestamp)
        if completed_incident:
            handle_incident_closure(completed_incident)

        # Update progress bar
        pct = min(1.0, frame_idx / total_frames)
        progress_bar.progress(pct)
        progress_caption.caption(
            t("progress_label", lang, current=frame_idx, total=total_frames, percent=int(pct * 100))
        )

    # Flush active incident at end of stream if any
    final_incident = event_manager.end_active_incident(timestamp=frame_idx / fps)
    if final_incident:
        handle_incident_closure(final_incident)

    cap.release()
    total_elapsed = time.time() - start_wall_time
    calc_fps = frame_idx / max(0.001, total_elapsed)

    st.success(t("playback_complete", lang, frames=frame_idx, duration=total_elapsed))

    # Summary performance metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(t("stats_frames_processed", lang), frame_idx)
    m2.metric(t("stats_fps", lang), f"{calc_fps:.1f}")
    m3.metric(t("stats_incidents_found", lang), len(detected_incidents))


# -----------------------------------------------------------------------------
# 6. Tab 2: Reports View
# -----------------------------------------------------------------------------
def render_reports_tab(lang: str) -> None:
    """Renders the Reports tab displaying master CSV logs, keyframes, AI dispatch, and export options.

    Args:
        lang: Active language ('en' or 'ar').
    """
    st.header(t("reports_header", lang))
    st.caption(t("reports_subtitle", lang))

    # Read master incidents log from CSV
    df = read_incidents_csv()

    if df.empty:
        st.info(t("no_reports", lang))
        return

    # Collision Class Filter
    classes = [t("all_classes", lang)] + sorted(list(df["class"].dropna().unique()))
    selected_class = st.selectbox(t("filter_by_class", lang), options=classes)

    filtered_df = df if selected_class == t("all_classes", lang) else df[df["class"] == selected_class]

    # Display Master Table
    st.subheader(t("reports_table_header", lang))
    st.dataframe(filtered_df, use_container_width=True)

    # Master CSV Export button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t("download_csv_btn", lang),
        data=csv_bytes,
        file_name="master_incidents_report.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Detailed Incident Cards & Keyframe Thumbnails
    for _, row in filtered_df.iterrows():
        inc_id = str(row["incident_id"])
        cls_raw = str(row["class"])
        cls_localized = translate_class_name(cls_raw, lang=lang)
        conf_val = float(row["max_confidence"])
        img_path = str(row["image_path"])
        start_ts = str(row.get("start_timestamp", ""))
        end_ts = str(row.get("end_timestamp", ""))

        with st.expander(f"🚨 {cls_localized} — {inc_id} ({start_ts})", expanded=False):
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.caption(t("keyframe_label", lang))
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning(f"Keyframe image missing at: {img_path}")

            with col_info:
                st.write(f"**ID:** `{inc_id}`")
                st.write(f"**Class:** {cls_localized}")
                st.write(f"**Confidence:** `{conf_val * 100:.1f}%`")
                st.write(f"**Start Time:** {start_ts}")
                st.write(f"**End Time:** {end_ts}")

                # AI Multimodal Dispatch Report Generation with Gemini
                dispatch_key = f"dispatch_{inc_id}"
                if dispatch_key not in st.session_state:
                    st.session_state[dispatch_key] = None

                btn_key = f"btn_gen_{inc_id}"
                if st.button(t("generate_dispatch_btn", lang), key=btn_key):
                    with st.spinner(t("dispatch_loading", lang)):
                        report_data = generate_dispatch_report(
                            image_path=img_path,
                            detections=cls_raw,
                            timestamp=start_ts,
                            lang=lang,
                        )
                        st.session_state[dispatch_key] = report_data
                        st.success(t("dispatch_success", lang))

                # Display generated dispatch assessment if available
                dispatch_data = st.session_state.get(dispatch_key)
                dispatch_rendered_text = None
                if dispatch_data:
                    st.markdown(f"#### {t('dispatch_report_title', lang)}")
                    dispatch_rendered_text = dispatch_data.get("rendered_text") or render_dispatch_text(dispatch_data, lang=lang)
                    st.code(dispatch_rendered_text, language="markdown")

                # Export & Dispatch Options
                col_pdf, col_alert = st.columns([1, 1])
                with col_pdf:
                    pdf_bytes = generate_report_pdf(
                        incident_id=inc_id,
                        class_name=cls_localized,
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        max_confidence=conf_val,
                        image_path=img_path,
                        dispatch_text=dispatch_rendered_text,
                        lang=lang,
                    )
                    st.download_button(
                        label=t("download_pdf_btn", lang),
                        data=pdf_bytes,
                        file_name=f"{inc_id}_report.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{inc_id}",
                        use_container_width=True,
                    )
                with col_alert:
                    if st.button(t("send_alert_btn", lang), key=f"alert_btn_{inc_id}", use_container_width=True):
                        disp_txt = dispatch_rendered_text or f"Incident: {cls_localized} ({inc_id})\nTime: {start_ts}"
                        res = send_incident_alert(
                            incident=row,
                            report_text=disp_txt,
                            image_path=img_path,
                            dry_run=st.session_state.get("dry_run_alerts", config.alerts_dry_run),
                        )
                        summary = ", ".join([f"{k}: {'OK' if v else 'FAIL'}" for k, v in res.items()])
                        st.success(t("alert_sent_success", lang, results=summary))


# -----------------------------------------------------------------------------
# 7. Tab 3: About View
# -----------------------------------------------------------------------------
def render_about_tab(lang: str) -> None:
    """Renders the About tab detailing architecture, classes, and specifications.

    Args:
        lang: Active language ('en' or 'ar').
    """
    st.header(t("about_title", lang))
    st.write(t("about_desc", lang))

    st.markdown("---")
    st.subheader(t("features_title", lang))
    st.markdown(t("feature_1", lang))
    st.markdown(t("feature_2", lang))
    st.markdown(t("feature_3", lang))
    st.markdown(t("feature_4", lang))
    st.markdown(t("feature_5", lang))

    st.markdown("---")
    st.subheader(t("classes_title", lang))

    # Display supported classes in a clean two-column grid
    col1, col2 = st.columns(2)
    classes_list = list(CLASS_NAMES)
    mid = len(classes_list) // 2

    with col1:
        for c in classes_list[:mid]:
            is_acc = c in ACCIDENT_CLASSES
            icon = "🚨" if is_acc else "🚗"
            st.write(f"{icon} **{c}** — *{translate_class_name(c, lang)}*")

    with col2:
        for c in classes_list[mid:]:
            is_acc = c in ACCIDENT_CLASSES
            icon = "🚨" if is_acc else "🚗"
            st.write(f"{icon} **{c}** — *{translate_class_name(c, lang)}*")

    st.markdown("---")
    st.subheader(t("tech_stack_title", lang))
    st.markdown(
        """
        - **Computer Vision**: Ultralytics YOLOv8 / YOLO11 (`best.pt`) & ByteTrack Tracking
        - **Multimodal AI**: Google GenAI SDK (`google-genai`, `gemini-2.5-flash`)
        - **Web Application**: Streamlit, Python-Bidi, Arabic-Reshaper
        - **Document Generation**: FPDF2 PDF & Pandas CSV Engine
        - **Compute Runtime**: Apple Silicon MPS / NVIDIA CUDA / CPU Auto-Detection
        """
    )


# -----------------------------------------------------------------------------
# 8. Main Application Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Controls the main application execution lifecycle."""
    # Render sidebar controls
    lang = render_sidebar()

    # Apply RTL CSS conditionally strictly when Arabic is active
    apply_rtl_css_if_arabic(lang)

    # Title Banner
    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    # Retrieve cached detector
    selected_model = st.session_state.get("selected_model", "best.pt")
    confidence_thresh = st.session_state.get("confidence_threshold", config.confidence_threshold)
    injury_path = "injury.pt" if st.session_state.get("use_injury", False) else None

    detector = get_cached_detector(
        model_path=selected_model,
        injury_model_path=injury_path,
        confidence_threshold=confidence_thresh,
        device=config.device,
    )

    # Main Navigation Tabs
    tab_live, tab_reports, tab_about = st.tabs([
        t("tab_live_analysis", lang),
        t("tab_reports", lang),
        t("tab_about", lang),
    ])

    with tab_live:
        render_live_analysis_tab(detector=detector, lang=lang)

    with tab_reports:
        render_reports_tab(lang=lang)

    with tab_about:
        render_about_tab(lang=lang)


if __name__ == "__main__":
    main()
