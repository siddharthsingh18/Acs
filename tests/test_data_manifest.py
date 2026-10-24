import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = PROJECT_ROOT / "ml/scripts/fetch_data.py"
SPEC = importlib.util.spec_from_file_location("acs_fetch_data", FETCH_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fetch_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_data)


def test_sample_hash_manifest_verification(tmp_path):
    paths = [tmp_path / f"sample-{index}.txt" for index in range(3)]
    entries = []
    for path in paths:
        content = f"sample content {path.stem}".encode()
        path.write_bytes(content)
        entries.append(f"{hashlib.sha256(content).hexdigest()}  {path.name}")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")

    assert fetch_data.verify_sample_hashes(manifest, sample_size=2) == 2

    paths[0].write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch_data.verify_sample_hashes(manifest, sample_size=3)


def test_enabled_source_requires_approval_and_pinned_archive():
    source = {
        "id": "candidate",
        "enabled": True,
        "status": "blocked",
        "provider": "url",
        "url": "https://example.invalid/archive.zip",
        "expected_file_count": None,
        "archive_sha256": None,
    }

    with pytest.raises(ValueError, match="status: approved"):
        fetch_data._validate_sources([source])


def test_enabled_source_requires_explicit_usage_rights_and_attribution():
    source = {
        "id": "candidate",
        "enabled": True,
        "status": "approved",
        "provider": "url",
        "url": "https://example.invalid/archive.zip",
        "expected_file_count": 1,
        "archive_sha256": "0" * 64,
        "redistribution_allowed": "unknown",
        "model_training_allowed": True,
        "license": "CC0",
        "attribution_text": "Dataset creator attribution.",
    }

    with pytest.raises(ValueError, match="verified redistribution and training permission"):
        fetch_data._validate_sources([source])


def test_checked_in_manifest_has_no_unapproved_enabled_sources():
    manifest = yaml.safe_load(fetch_data.MANIFEST_PATH.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="No dataset sources are approved and enabled"):
        fetch_data._validate_sources(manifest["sources"])
