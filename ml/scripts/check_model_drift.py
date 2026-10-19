#!/usr/bin/env python3
"""
ml/scripts/check_model_drift.py
CLI tool and cron job for weekly model drift analysis across all CCTV cameras.

Outputs:
- Console summary table with PSI scores, detection count shifts, and diagnosis.
- Generates markdown report: reports/model_drift_report.md.
- Dispatches emergency alert if sharp drift is detected.
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.drift_service import drift_service


def main():
    parser = argparse.ArgumentParser(description="Weekly CCTV Model Drift & Shift Detection")
    parser.add_argument("--window-days", type=int, default=7, help="Rolling evaluation window in days (default: 7)")
    parser.add_argument("--camera-id", type=str, default=None, help="Check specific camera ID only")
    parser.add_argument("--output-md", type=str, default="reports/model_drift_report.md", help="Markdown report path")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if any camera drifted")
    args = parser.parse_args()

    print("=" * 80)
    print("ACS CAMERA MODEL DRIFT & ANOMALY DETECTOR")
    print(f"Window: past {args.window_days} days | Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    if args.camera_id:
        reports = [drift_service.check_camera_drift(args.camera_id, window_days=args.window_days)]
    else:
        reports = drift_service.check_all_cameras()

    if not reports:
        print("No cameras configured or found in database.")
        sys.exit(0)

    print("\n{:<22} {:<22} {:<10} {:<12} {:<10} {:<10}".format(
        "Camera ID", "Camera Name", "PSI", "Count Delta", "Severity", "Drift?"
    ))
    print("-" * 90)

    critical_count = 0
    warning_count = 0

    for r in reports:
        is_drift_str = "YES 🚨" if r.drift_detected else "NO  ✅"
        print("{:<22} {:<22} {:<10.3f} {:<+11.1%} {:<10} {:<10}".format(
            r.camera_id[:20],
            r.camera_name[:20],
            r.psi_score,
            r.count_delta_ratio,
            r.drift_severity.upper(),
            is_drift_str,
        ))
        if r.drift_severity == "critical":
            critical_count += 1
            for reason in r.reasons:
                print(f"   ⚠️  REASON: {reason}")
            print(f"   💡 ACTION: {r.suggested_action}\n")
        elif r.drift_severity == "warning":
            warning_count += 1

    # Generate Markdown Report
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Weekly Model Drift & CCTV Camera Anomaly Report\n\n")
        f.write(f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"- **Evaluation Window**: {args.window_days} days\n")
        f.write(f"- **Total Cameras Evaluated**: {len(reports)}\n")
        f.write(f"- **Critical Drifts**: {critical_count}\n")
        f.write(f"- **Warnings**: {warning_count}\n\n")

        f.write("| Camera ID | Name | PSI Score | Count Shift | Status | Drift Detected | Suggested Action |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in reports:
            drift_icon = "🚨 YES" if r.drift_detected else "✅ NO"
            f.write(f"| `{r.camera_id}` | {r.camera_name} | `{r.psi_score:.3f}` | `{r.count_delta_ratio * 100:+.1f}%` | **{r.drift_severity.upper()}** | {drift_icon} | {r.suggested_action} |\n")

        if critical_count > 0:
            f.write("\n## Detailed Anomaly Diagnoses\n\n")
            for r in reports:
                if r.drift_detected:
                    f.write(f"### Camera `{r.camera_id}` ({r.camera_name})\n")
                    for reason in r.reasons:
                        f.write(f"- {reason}\n")
                    f.write(f"- **Remediation**: {r.suggested_action}\n\n")

    print(f"\nMarkdown report generated at: {args.output_md}")

    if args.fail_on_drift and critical_count > 0:
        print(f"\nFAILURE: {critical_count} camera(s) exhibit sharp model drift.")
        sys.exit(1)


if __name__ == "__main__":
    main()
