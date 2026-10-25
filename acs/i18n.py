"""
acs.i18n
Internationalization and localization module providing UI strings
and class name translations for English (en) and Arabic (ar).
"""
from typing import Dict, Any

# Comprehensive UI text catalog for English and Arabic
UI_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        # App identity & layout
        "app_title": "CCTV Accident Detection & Dispatch System",
        "app_subtitle": "Enterprise Automated Surveillance, Tracking & Emergency Dispatch",
        "sidebar_title": "ACS Control Center",
        "lang_select": "Language / اللغة",
        "model_select": "Primary YOLO Model (.pt)",
        "confidence_slider": "Detection Confidence Threshold",
        "use_injury_checkbox": "Enable Secondary Injury Detection",
        "injury_disabled_notice": "injury.pt not found — Injury model disabled",
        "device_badge": "Compute Device: {device}",
        "active_settings_header": "System Settings",
        "alerts_section_title": "Emergency Alert Channels",
        "enable_alerts_toggle": "Enable Emergency Alerts",
        "dry_run_toggle": "Alerts Dry-Run Mode (Demo)",
        "send_test_alert_btn": "📨 Send Test Alert",
        "test_alert_success": "Test alert dispatched successfully ({results})",
        "test_alert_info": "Test alert processed ({results})",
        "alert_dispatched_toast": "🚨 Dispatch alert sent for {incident_id} via {channels}",
        "send_alert_btn": "📨 Dispatch Emergency Alert",
        "alert_sent_success": "Emergency alert sent successfully ({results})",

        # Tabs
        "tab_live_analysis": "🎥 Live Analysis",
        "tab_reports": "📋 Incident Reports",
        "tab_about": "ℹ️ About",

        # Live Analysis
        "analysis_header": "Automated CCTV Video Analysis & Incident Tracking",
        "analysis_subtitle": "Upload video footage or select a sample clip to run real-time ByteTrack tracking and incident aggregation.",
        "video_source_label": "Video Source",
        "upload_option": "Upload Video File",
        "sample_option": "Sample Video (acc_ex.mp4)",
        "upload_label": "Upload video file (MP4, AVI, MOV, MKV)",
        "start_analysis_btn": "▶️ Run Detection & Tracking",
        "stop_analysis_btn": "⏹️ Stop Analysis",
        "processing_video": "Processing video stream with ByteTrack...",
        "video_player_col": "Annotated Video Stream",
        "incident_feed_col": "Live Incident Feed",
        "no_incidents_yet": "No active or recorded incidents detected yet.",
        "incident_detected_toast": "🚨 Incident Detected: {incident_id} ({class_name})",
        "class_detected_info": "{class_name} detected",
        "playback_complete": "Analysis complete. Processed {frames} frames in {duration:.1f}s.",
        "progress_label": "Analysis Progress: Frame {current}/{total} ({percent}%)",
        "stats_frames_processed": "Frames Analyzed",
        "stats_fps": "Inference FPS",
        "stats_incidents_found": "Total Incidents",
        "incident_card_title": "{class_name}",
        "confidence_label": "Confidence: {confidence:.1%}",
        "timestamp_label": "Time: {timestamp}",
        "duration_label": "Duration: {duration:.1f}s",

        # Reports
        "reports_header": "Recorded Accident Incidents & Dispatch Reports",
        "reports_subtitle": "Review detected incident keyframes, generate AI dispatch assessments with Gemini, and export reports as PDF or CSV.",
        "no_reports": "No incidents recorded in the system yet. Run Live Analysis to generate incident data.",
        "reports_table_header": "Master Incidents Log",
        "filter_by_class": "Filter by Collision Class",
        "all_classes": "All Classes",
        "keyframe_label": "Keyframe Image",
        "generate_dispatch_btn": "⚡ Generate Dispatch Assessment",
        "dispatch_loading": "Generating multimodal emergency dispatch assessment with Gemini...",
        "dispatch_success": "Dispatch assessment generated successfully!",
        "dispatch_report_title": "Emergency Dispatch Report",
        "download_csv_btn": "📥 Download Master CSV Log",
        "download_pdf_btn": "📄 Download PDF Report",
        "export_header": "Export Options",

        # About
        "about_title": "Automated CCTV Accident Detection System (ACS)",
        "about_desc": "ACS is an enterprise-grade AI system designed for municipal traffic surveillance, real-time vehicle accident detection, ByteTrack tracking, multi-frame keyframe capture, and automated 911 dispatch multimodal reporting with Google Gemini.",
        "features_title": "Core System Architecture",
        "feature_1": "• **YOLO Vision Inference**: Deep learning vision models trained on traffic accidents and collisions.",
        "feature_2": "• **ByteTrack Multi-Object Tracking**: Trajectory tracking with persist=True preventing duplicate counts.",
        "feature_3": "• **Accident Event Manager**: Temporal state machine requiring 5 consecutive frames, peak keyframe retention, 3s inactivity timeout, and 10s cooldown.",
        "feature_4": "• **Multimodal Gemini Dispatch**: Generates structured emergency response reports with vehicle identification, severity rating, and recommended units (EMS, police, fire, tow).",
        "feature_5": "• **Intuitive Operator Dashboard**: Clean, streamlined layout for real-time monitoring and incident review.",
        "classes_title": "Supported Vision Classes",
        "tech_stack_title": "Technology Stack",
    },
    "ar": {
        # App identity & layout
        "app_title": "نظام كشف الحوادث وإرسال الطوارئ عبر كاميرات المراقبة",
        "app_subtitle": "منظومة المراقبة المرورية الذكية والتتبع وتوليد تقارير الطوارئ",
        "sidebar_title": "لوحة تحكم النظام (ACS)",
        "lang_select": "اللغة / Language",
        "model_select": "نموذج YOLO الأساسي (.pt)",
        "confidence_slider": "حد ثقة الاكتشاف",
        "use_injury_checkbox": "تفعيل نموذج الإصابات الثانوي",
        "injury_disabled_notice": "لم يتم العثور على injury.pt — نموذج الإصابات معطل",
        "device_badge": "جهاز المعالجة: {device}",
        "active_settings_header": "إعدادات المنظومة",
        "alerts_section_title": "قنوات تنبيه الطوارئ",
        "enable_alerts_toggle": "تفعيل تنبيهات الطوارئ",
        "dry_run_toggle": "وضع المحاكاة (تجريبي)",
        "send_test_alert_btn": "📨 إرسال تنبيه تجريبي",
        "test_alert_success": "تم إرسال التنبيه التجريبي بنجاح ({results})",
        "test_alert_info": "تمت معالجة التنبيه التجريبي ({results})",
        "alert_dispatched_toast": "🚨 تم إرسال تنبيه للحادث {incident_id} عبر {channels}",
        "send_alert_btn": "📨 إرسال تنبيه الطوارئ",
        "alert_sent_success": "تم إرسال تنبيه الطوارئ بنجاح ({results})",

        # Tabs
        "tab_live_analysis": "🎥 التحليل المباشر",
        "tab_reports": "📋 تقارير الحوادث",
        "tab_about": "ℹ️ حول المنظومة",

        # Live Analysis
        "analysis_header": "التحليل التلقائي لكاميرات المراقبة وتتبع الحوادث",
        "analysis_subtitle": "ارفع مقطع فيديو أو اختر المقطع التجريبي لتشغيل تتبع ByteTrack وتجميع الحوادث في الوقت الفعلي.",
        "video_source_label": "مصدر الفيديو",
        "upload_option": "رفع ملف فيديو",
        "sample_option": "فيديو تجريبي (acc_ex.mp4)",
        "upload_label": "ارفع مقطع الفيديو (MP4, AVI, MOV, MKV)",
        "start_analysis_btn": "▶️ تشغيل الكشف والتتبع",
        "stop_analysis_btn": "⏹️ إيقاف التحليل",
        "processing_video": "جاري معالجة إطارات الفيديو مع تتبع ByteTrack...",
        "video_player_col": "البث المرئي المعالج",
        "incident_feed_col": "سجل الحوادث المباشر",
        "no_incidents_yet": "لم يتم تسجيل أي حوادث في هذا المقطع حتى الآن.",
        "incident_detected_toast": "🚨 تم اكتشاف حادث: {incident_id} ({class_name})",
        "class_detected_info": "تم اكتشاف {class_name}",
        "playback_complete": "اكتمل التحليل. تمت معالجة {frames} إطار خلال {duration:.1f} ثانية.",
        "progress_label": "تقدم التحليل: الإطار {current}/{total} ({percent}%)",
        "stats_frames_processed": "الإطارات المعالجة",
        "stats_fps": "سرعة المعالجة (FPS)",
        "stats_incidents_found": "إجمالي الحوادث",
        "incident_card_title": "{class_name}",
        "confidence_label": "نسبة الثقة: {confidence:.1%}",
        "timestamp_label": "الوقت: {timestamp}",
        "duration_label": "المدة: {duration:.1f} ثانية",

        # Reports
        "reports_header": "سجل الحوادث المسجلة وتقارير إرسال الطوارئ",
        "reports_subtitle": "استعرض الإطارات الرئيسية للحوادث، وقم بتوليد تقارير إرسال الطوارئ بواسطة Gemini، وتصدير التقارير كـ PDF أو CSV.",
        "no_reports": "لا توجد حوادث مسجلة في النظام حتى الآن. قم بتشغيل التحليل المباشر لإنشاء بيانات الحوادث.",
        "reports_table_header": "جدول الحوادث الرئيسي",
        "filter_by_class": "تصفية حسب نوع الحادث",
        "all_classes": "جميع الفئات",
        "keyframe_label": "الإطار الرئيسي للحادث",
        "generate_dispatch_btn": "⚡ توليد تقرير الإرسال الذكي",
        "dispatch_loading": "جاري تحليل الحادث وتوليد تقرير إرسال الطوارئ بواسطة Gemini...",
        "dispatch_success": "تم إعداد تقرير إرسال الطوارئ بنجاح!",
        "dispatch_report_title": "تقرير إرسال الطوارئ المعتمد",
        "download_csv_btn": "📥 تنزيل سجل الحوادث (CSV)",
        "download_pdf_btn": "📄 تنزيل تقرير الحادث (PDF)",
        "export_header": "خيارات التصدير",

        # About
        "about_title": "نظام كشف الحوادث المرورية الذكي عبر كاميرات المراقبة (ACS)",
        "about_desc": "ACS هو نظام ذكاء اصطناعي متطور مخصص لمراقبة الحركة المرورية في المدن، واكتشاف حوادث المركبات في الوقت الفعلي، وتتبع الأجسام باستخدام ByteTrack، وحفظ الإطارات الرئيسية، وتوليد تقارير طوارئ ذكية بالتكامل مع Google Gemini.",
        "features_title": "البنية التقنية والميزات الأساسية",
        "feature_1": "• **كشف المركبات والحوادث بـ YOLO**: نماذج دقيقة قادرة على تصنيف 9 فئات مرورية وحوادث تصادم.",
        "feature_2": "• **تتبع الأجسام المتقدم (ByteTrack)**: تتبع مسار المركبات مع حفظ المعرفات لتجنب التكرار.",
        "feature_3": "• **إدارة دورة حياة الحادث (Event Manager)**: خوارزمية تجميع زمني تشترط 5 إطارات متتالية وتلتقط أعلى إطار ثقة وتنهي الحادث بعد 3 ثوانٍ وتطبق فترة تهدئة 10 ثوانٍ.",
        "feature_4": "• **تقارير إرسال طوارئ متعددة الوسائط (Gemini)**: تقييم ذكي ومفصل يحدد المركبات، نقطة الاصطدام، درجة الخطورة، والوحدات المطلوبة (إسعاف، شرطة، إطفاء، سحب).",
        "feature_5": "• **واجهة ثنائية اللغة وتخطيط RTL**: دعم كامل للغتين العربية والإنجليزية مع تطبيق فوري لتخطيط اليمين لليسار.",
        "classes_title": "فئات الرؤية الحاسوبية المدعومة",
        "tech_stack_title": "التقنيات المستخدمة",
    },
}

# Translation mappings for model class names
CLASS_TRANSLATIONS: Dict[str, str] = {
    "bike": "دراجة",
    "car": "سيارة",
    "person": "شخص",
    "injured": "مصاب",
    "vehicle": "مركبة",
    "accident": "حادث",
}


def t(key: str, lang: str = "en", **kwargs: Any) -> str:
    """Retrieves localized UI text string for a given key and language.

    Args:
        key: UI string key.
        lang: Selected language code ('en' or 'ar').
        **kwargs: Optional string formatting arguments.

    Returns:
        str: Localized and formatted UI string.
    """
    lang_dict = UI_STRINGS.get(lang, UI_STRINGS["en"])
    template = lang_dict.get(key, UI_STRINGS["en"].get(key, key))
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def translate_class_name(class_name: str, lang: str = "en") -> str:
    """Translates a model class name to the requested language.

    Args:
        class_name: Raw YOLO class name.
        lang: Target language ('en' returns raw class, 'ar' returns translated Arabic).

    Returns:
        str: Translated class name.
    """
    if lang != "ar":
        return class_name

    normalized = str(class_name).lower().strip()
    if normalized in CLASS_TRANSLATIONS:
        return CLASS_TRANSLATIONS[normalized]

    # Split and map individual words if compound
    words = normalized.split("_")
    translated = [CLASS_TRANSLATIONS.get(w, w) for w in words]
    if "accident" in normalized:
        return "حادث " + " ".join([w for w in translated if w != "accident"])

    return " ".join(translated)
