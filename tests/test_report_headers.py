from pathlib import Path

from ml.eval.report_headers import format_header, make_header
from ml.eval.check_report_headers import verify_latest


ROOT = Path(__file__).resolve().parents[1]


def test_latest_report_headers_match_current_hashes(tmp_path: Path):
    model = ROOT / "best.pt"
    manifest = ROOT / "ml/eval/data_manifest.yaml"
    header = make_header(model, manifest)
    reports_root = tmp_path / "reports"
    run_dir = reports_root / "runs" / header["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.md").write_text(
        format_header(header, {"test_clips": 0}) + "# Metrics\n",
        encoding="utf-8",
    )
    (reports_root / "latest_run.txt").write_text(header["run_id"] + "\n", encoding="utf-8")

    assert verify_latest(model, manifest, reports_root) == []


def test_report_hash_mismatch_fails_closed(tmp_path: Path):
    model = ROOT / "best.pt"
    manifest = ROOT / "ml/eval/data_manifest.yaml"
    header = make_header(model, manifest)
    header["model_sha256"] = "0" * 64
    reports_root = tmp_path / "reports"
    run_dir = reports_root / "runs" / header["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.md").write_text(
        format_header(header, {}) + "# Metrics\n",
        encoding="utf-8",
    )
    (reports_root / "latest_run.txt").write_text(header["run_id"] + "\n", encoding="utf-8")

    assert any(
        "model_sha256 does not match" in error
        for error in verify_latest(model, manifest, reports_root)
    )
