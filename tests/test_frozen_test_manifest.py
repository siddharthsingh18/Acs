import hashlib
from pathlib import Path

from ml.scripts.verify_test_manifest import verify_test_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_test_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    image = data_root / "merged/images/test/camera/clip.jpg"
    label = data_root / "merged/labels/test/camera/clip.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image bytes")
    label.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = data_root / "test_manifest.sha256"
    manifest.write_text(
        "# frozen\n"
        f"{_digest(image)}  merged/images/test/camera/clip.jpg\n"
        f"{_digest(label)}  merged/labels/test/camera/clip.txt\n",
        encoding="utf-8",
    )
    return data_root, manifest


def test_frozen_test_manifest_accepts_unchanged_files(tmp_path: Path):
    data_root, manifest = _build_test_tree(tmp_path)
    assert verify_test_manifest(manifest, data_root) == []


def test_frozen_test_manifest_rejects_modified_or_added_files(tmp_path: Path):
    data_root, manifest = _build_test_tree(tmp_path)
    (data_root / "merged/images/test/camera/clip.jpg").write_bytes(b"changed")
    (data_root / "merged/images/test/camera/new.jpg").write_bytes(b"new")
    errors = verify_test_manifest(manifest, data_root)
    assert any("SHA256 changed" in error for error in errors)
    assert any("unmanifested frozen test file" in error for error in errors)


def test_frozen_test_manifest_fails_closed_if_missing(tmp_path: Path):
    assert "Frozen test manifest is missing" in verify_test_manifest(tmp_path / "missing.sha256", tmp_path)[0]
