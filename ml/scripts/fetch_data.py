#!/usr/bin/env python3
"""Fetch only explicitly approved, checksum-pinned dataset archives."""
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List

import yaml
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "ml/configs/data_sources.yaml"
DATA_ROOT = PROJECT_ROOT / "ml/data"
HASH_MANIFEST_PATH = DATA_ROOT / "manifest.sha256"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sample_hashes(manifest_path: Path, sample_size: int = 5) -> int:
    """Verify up to sample_size sorted entries in a sha256sum-compatible manifest."""
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative_path = line.split(maxsplit=1)
        entries.append((digest, relative_path.lstrip("*")))

    sample = sorted(entries, key=lambda item: item[1])[:sample_size]
    if not sample:
        return 0

    root = manifest_path.parent
    for expected, relative_path in sample:
        file_path = root / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Hash-manifest file is missing: {relative_path}")
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {relative_path}: expected {expected}, got {actual}")
    return len(sample)


def _dotenv_credentials() -> Dict[str, str]:
    values = dotenv_values(PROJECT_ROOT / ".env")
    return {key: value for key, value in values.items() if value}


def _archive_url(source: Dict[str, Any], credentials: Dict[str, str]) -> str:
    provider = source["provider"]
    if provider == "url":
        return str(source["url"])
    if provider == "huggingface":
        repo_id = urllib.parse.quote(str(source["repo_id"]), safe="/")
        revision = urllib.parse.quote(str(source["revision"]), safe="")
        filename = urllib.parse.quote(str(source["filename"]), safe="/")
        return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    if provider == "roboflow":
        api_key = credentials.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY must be set in the repository .env file.")
        workspace = urllib.parse.quote(str(source["workspace"]), safe="")
        project = urllib.parse.quote(str(source["project"]), safe="")
        version = int(source["version"])
        query = urllib.parse.urlencode({"api_key": api_key, "format": source.get("format", "yolov8")})
        return f"https://api.roboflow.com/{workspace}/{project}/{version}?{query}"
    raise ValueError(f"Unsupported dataset provider: {provider}")


def _download(url: str, destination: Path, headers: Dict[str, str]) -> None:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Dataset download failed with HTTP status {error.code}.") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Dataset download failed: {error.reason}") from error


def _resolve_roboflow_export(api_url: str) -> str:
    try:
        with urllib.request.urlopen(api_url, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Roboflow export lookup failed with HTTP status {error.code}.") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError("Roboflow did not return a usable export for this manifest entry.")
    export = payload.get("export")
    download_url = export.get("link") if isinstance(export, dict) else payload.get("url")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError("Roboflow response did not include a valid HTTPS archive URL.")
    return download_url


def _validate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enabled = [source for source in sources if source.get("enabled") is True]
    if not enabled:
        blocked = [source.get("id", "<unnamed>") for source in sources]
        raise RuntimeError(
            "No dataset sources are approved and enabled. Resolve license, file-count, and SHA256 "
            f"blockers in {MANIFEST_PATH.relative_to(PROJECT_ROOT)} before fetching. Listed sources: "
            + ", ".join(blocked)
        )
    for source in enabled:
        if source.get("status") != "approved":
            raise ValueError(f"Enabled source {source.get('id')} must have status: approved.")
        if source.get("redistribution_allowed") is not True or source.get("model_training_allowed") is not True:
            raise ValueError(f"Enabled source {source.get('id')} must have verified redistribution and training permission.")
        if not source.get("license") or not source.get("attribution_text"):
            raise ValueError(f"Enabled source {source.get('id')} needs verified license and attribution_text.")
        if not isinstance(source.get("expected_file_count"), int) or source["expected_file_count"] <= 0:
            raise ValueError(f"Source {source.get('id')} needs a positive expected_file_count.")
        if not isinstance(source.get("archive_sha256"), str) or not SHA256_PATTERN.fullmatch(source["archive_sha256"]):
            raise ValueError(f"Source {source.get('id')} needs a pinned lowercase archive_sha256.")
        if source.get("provider") == "url" and not source.get("url"):
            raise ValueError(f"Source {source.get('id')} needs a download URL.")
        if source.get("provider") == "huggingface" and not all(source.get(key) for key in ("repo_id", "revision", "filename")):
            raise ValueError(f"Source {source.get('id')} needs repo_id, revision, and filename.")
        if source.get("provider") == "roboflow" and not all(source.get(key) for key in ("workspace", "project", "version")):
            raise ValueError(f"Source {source.get('id')} needs workspace, project, and version.")
    return enabled


def _extract_archive(archive_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        for item in files:
            member = PurePosixPath(item.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Archive contains unsafe path: {item.filename}")
            target = (destination / Path(*member.parts)).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive member escapes destination: {item.filename}")
        archive.extractall(destination)
    return len(files)


def _write_hash_manifest() -> None:
    lines = []
    for file_path in sorted(DATA_ROOT.rglob("*")):
        if file_path.is_file() and file_path != HASH_MANIFEST_PATH:
            relative = file_path.relative_to(DATA_ROOT).as_posix()
            lines.append(f"{sha256_file(file_path)}  {relative}")
    HASH_MANIFEST_PATH.write_text(
        "# Generated by ml/scripts/fetch_data.py; commit this file with the dataset version.\n"
        + "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )
    verify_sample_hashes(HASH_MANIFEST_PATH)


def fetch_sources(sources: Iterable[Dict[str, Any]]) -> None:
    credentials = _dotenv_credentials()
    hf_token = credentials.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    for source in sources:
        source_id = source["id"]
        download_url = _archive_url(source, credentials)
        if source["provider"] == "roboflow":
            download_url = _resolve_roboflow_export(download_url)

        source_dir = DATA_ROOT / "raw" / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        archive_path = source_dir / "source.zip"
        temporary_archive = source_dir / "source.zip.part"
        _download(download_url, temporary_archive, headers)
        actual_sha256 = sha256_file(temporary_archive)
        if actual_sha256 != source["archive_sha256"]:
            temporary_archive.unlink(missing_ok=True)
            raise ValueError(
                f"Archive SHA256 mismatch for {source_id}: expected {source['archive_sha256']}, "
                f"got {actual_sha256}"
            )
        temporary_archive.replace(archive_path)
        file_count = _extract_archive(archive_path, source_dir / "extracted")
        if file_count != source["expected_file_count"]:
            raise ValueError(
                f"Extracted file count mismatch for {source_id}: expected "
                f"{source['expected_file_count']}, got {file_count}"
            )
        print(f"Fetched {source_id}: verified {file_count} files and archive SHA256.")

    _write_hash_manifest()


def main() -> None:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list):
        raise ValueError(f"Expected a sources list in {MANIFEST_PATH}")
    fetch_sources(_validate_sources(sources))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Data acquisition failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
