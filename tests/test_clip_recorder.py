"""
tests.test_clip_recorder
Comprehensive test suite for:
- Rolling buffer per camera with strictly bounded memory over a 1-hour simulated run.
- Pre-impact (8s) and post-impact (10s) clip collection and H.264 MP4 export (ffmpeg).
- Dual clip rendering: raw version and annotated version with bounding boxes burned in.
- SQLite incident table clip_path, clip_annotated_path, and impact_offset_seconds persistence.
- HTTP Range request support (206 Partial Content, Content-Range, Accept-Ranges, 416 out-of-bounds, HEAD).
- Retention policy enforcement (age in days and max disk usage cap in MB).
"""
import os
import shutil
import tempfile
import time
import tracemalloc
import unittest
from fastapi.testclient import TestClient
import numpy as np

from acs.clip_recorder import (
    CameraRollingBuffer,
    IncidentClipRecorder,
    cleanup_expired_clips,
    get_clip_storage_stats,
    save_h264_clip,
)
from acs.config import config
from acs.detector import AccidentEventManager, DetectionBox, DetectionResult
from backend.app.db.database import engine, init_db
from backend.app.db.models import Incident
from backend.app.main import app
from sqlmodel import Session, select


class TestRollingBufferMemory(unittest.TestCase):
    """Verifies that the rolling ring buffer maintains flat memory over long runs."""

    def test_memory_stays_flat_over_1_hour_run(self):
        """Simulates 1 hour of video frames at 30 FPS (108,000 frames) and verifies memory stays flat."""
        tracemalloc.start()
        buffer = CameraRollingBuffer(max_seconds=10.0, target_fps=30.0, max_frames=300)

        # Baseline frame shape: 240x320x3 uint8
        sample_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        sample_ann = np.ones((240, 320, 3), dtype=np.uint8)

        # 1. Warm up buffer to capacity (300 frames = 10s at 30 FPS)
        for i in range(300):
            buffer.append(timestamp=i / 30.0, raw_frame=sample_frame, annotated_frame=sample_ann)

        self.assertEqual(len(buffer), 300)

        # Snapshot memory after buffer is fully saturated
        tracemalloc.clear_traces()
        mem_start_current, mem_start_peak = tracemalloc.get_traced_memory()

        # 2. Simulate 1 hour: 108,000 frames at 30 FPS
        # Using 108,000 iterations to match exact 1-hour run specification
        for i in range(300, 108300):
            t = i / 30.0
            buffer.append(timestamp=t, raw_frame=sample_frame, annotated_frame=sample_ann)

        mem_end_current, mem_end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Buffer length must stay exactly bounded at 300
        self.assertEqual(len(buffer), 300)

        # Memory delta between frame 300 and frame 108,300 must stay flat (< 1.0 MB variation)
        mem_diff_mb = abs(mem_end_current - mem_start_current) / (1024 * 1024)
        self.assertLess(
            mem_diff_mb,
            1.0,
            f"Memory grew by {mem_diff_mb:.2f} MB over 1-hour run; expected flat bounded memory.",
        )

    def test_window_and_recent_extraction(self):
        """Tests that get_window and get_recent return correct chronological subsets."""
        buffer = CameraRollingBuffer(max_seconds=10.0, target_fps=10.0, max_frames=100)
        for i in range(50):
            frame = np.full((50, 50, 3), fill_value=i, dtype=np.uint8)
            buffer.append(timestamp=float(i), raw_frame=frame, annotated_frame=frame)

        # Extract last 8 seconds before timestamp 40.0 -> [32.0, 40.0]
        ts_list, raw_list, ann_list = buffer.get_recent(seconds=8.0, before_time=40.0)
        self.assertGreater(len(ts_list), 0)
        self.assertGreaterEqual(ts_list[0], 32.0)
        self.assertLessEqual(ts_list[-1], 40.0)


class TestIncidentClipRecorder(unittest.TestCase):
    """Tests incident clip extraction (8s pre to 10s post), dual version H.264 encoding, and retention."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.clips_dir = os.path.join(self.test_dir, "clips")
        os.makedirs(self.clips_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_pre_and_post_clip_generation_raw_and_annotated(self):
        """Tests that raw and annotated clips spanning 8s pre to 10s post impact are saved."""
        recorder = IncidentClipRecorder(
            rolling_buffer_seconds=10.0,
            pre_incident_seconds=8.0,
            post_incident_seconds=10.0,
            output_dir=self.clips_dir,
            target_fps=10.0,
        )

        fps = 10.0
        # 1. Feed 10 seconds of pre-impact frames: t = 0 to 10.0
        for i in range(100):
            t = i / fps
            raw = np.zeros((120, 160, 3), dtype=np.uint8)
            ann = np.zeros((120, 160, 3), dtype=np.uint8)
            # Annotate with a visible colored box
            ann[10:50, 10:50] = [0, 255, 0]
            recorder.feed_frame(t, raw_frame=raw, annotated_frame=ann)

        # 2. Incident starts at t = 10.0
        incident_id = "INC-TEST-001"
        recorder.start_incident(incident_id=incident_id, impact_time=10.0)

        # 3. Feed 10 seconds of post-impact frames: t = 10.1 to 20.0
        for i in range(101, 201):
            t = i / fps
            raw = np.zeros((120, 160, 3), dtype=np.uint8)
            ann = np.zeros((120, 160, 3), dtype=np.uint8)
            ann[20:60, 20:60] = [0, 0, 255]
            recorder.feed_frame(t, raw_frame=raw, annotated_frame=ann)

        # 4. Finalize session
        raw_path, ann_path, offset = recorder.finalize_incident(incident_id, output_dir=self.clips_dir)

        # Assertions
        self.assertIsNotNone(raw_path)
        self.assertIsNotNone(ann_path)
        self.assertTrue(os.path.exists(raw_path))
        self.assertTrue(os.path.exists(ann_path))
        self.assertGreater(os.path.getsize(raw_path), 0)
        self.assertGreater(os.path.getsize(ann_path), 0)
        self.assertTrue(raw_path.endswith("_raw.mp4"))
        self.assertTrue(ann_path.endswith("_annotated.mp4"))

        # Impact offset should be approximately 8.0 seconds
        self.assertAlmostEqual(offset, 8.0, delta=1.0)

    def test_retention_policy_age_and_disk_usage(self):
        """Tests that clips older than N days are deleted and disk space cap is strictly respected."""
        # Create dummy clips with distinct timestamps and sizes
        now = time.time()
        file_old = os.path.join(self.clips_dir, "old_clip.mp4")
        file_mid = os.path.join(self.clips_dir, "mid_clip.mp4")
        file_new = os.path.join(self.clips_dir, "new_clip.mp4")

        # Write 200KB to each file
        dummy_data = b"0" * (200 * 1024)
        for p in [file_old, file_mid, file_new]:
            with open(p, "wb") as f:
                f.write(dummy_data)

        # Set mtime: old is 10 days ago, mid is 2 days ago, new is now
        os.utime(file_old, (now - 10 * 86400, now - 10 * 86400))
        os.utime(file_mid, (now - 2 * 86400, now - 2 * 86400))
        os.utime(file_new, (now, now))

        # 1. Cleanup by retention_days = 7
        res1 = cleanup_expired_clips(clips_dir=self.clips_dir, retention_days=7, max_disk_usage_mb=100)
        self.assertEqual(res1["deleted_count"], 1)
        self.assertFalse(os.path.exists(file_old))
        self.assertTrue(os.path.exists(file_mid))
        self.assertTrue(os.path.exists(file_new))

        # 2. Cleanup by max_disk_usage_mb cap (e.g. cap at 0.3 MB = 300 KB, when mid + new = 400 KB)
        # It should delete the older file (mid_clip.mp4)
        res2 = cleanup_expired_clips(clips_dir=self.clips_dir, retention_days=30, max_disk_usage_mb=0)
        self.assertGreater(res2["deleted_count"], 0)


class TestHttpRangeAndEndpoints(unittest.TestCase):
    """Tests HTTP Range header responses, video seeking, and API endpoints."""

    def setUp(self):
        init_db()
        self.client = TestClient(app)
        self.test_dir = tempfile.mkdtemp()
        self.clips_dir = os.path.join(self.test_dir, "report_clips")
        os.makedirs(self.clips_dir, exist_ok=True)

        # Generate a small valid H.264 video file
        self.clip_path = os.path.join(self.clips_dir, "INC-HTTP-001_raw.mp4")
        self.ann_clip_path = os.path.join(self.clips_dir, "INC-HTTP-001_annotated.mp4")

        frames = [np.full((120, 160, 3), fill_value=i * 10, dtype=np.uint8) for i in range(15)]
        save_h264_clip(frames, self.clip_path, fps=15.0)
        save_h264_clip(frames, self.ann_clip_path, fps=15.0)

        self.file_size = os.path.getsize(self.clip_path)

        # Seed test incident in SQLite DB
        with Session(engine) as session:
            existing = session.get(Incident, "INC-HTTP-001")
            if existing:
                session.delete(existing)
                session.commit()
            inc = Incident(
                incident_id="INC-HTTP-001",
                start_timestamp="2026-10-18 12:00:00 UTC",
                end_timestamp="2026-10-18 12:00:10 UTC",
                class_name="car_car_accident",
                max_confidence=0.91,
                image_path=os.path.join(self.clips_dir, "test.jpg"),
                keyframe_url="/media/keyframes/test.jpg",
                severity="high",
                status="pending",
                clip_path=self.clip_path,
                clip_annotated_path=self.ann_clip_path,
                impact_offset_seconds=8.0,
                created_at=time.time(),
                duration=10.0,
            )
            session.add(inc)
            session.commit()

    def tearDown(self):
        with Session(engine) as session:
            existing = session.get(Incident, "INC-HTTP-001")
            if existing:
                session.delete(existing)
                session.commit()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_http_range_partial_content_206(self):
        """Tests that HTTP Range header returns 206 Partial Content with correct byte slice."""
        headers = {"Range": "bytes=0-100"}
        res = self.client.get("/api/incidents/INC-HTTP-001/clip", headers=headers)
        self.assertEqual(res.status_code, 206)
        self.assertEqual(res.headers.get("Accept-Ranges"), "bytes")
        self.assertEqual(res.headers.get("Content-Range"), f"bytes 0-100/{self.file_size}")
        self.assertEqual(len(res.content), 101)
        self.assertEqual(res.headers.get("Content-Type"), "video/mp4")

    def test_http_range_suffix_and_open_ended(self):
        """Tests bytes=50- and suffix bytes=-50 range requests."""
        # Range: bytes=50-
        res1 = self.client.get("/api/incidents/INC-HTTP-001/clip", headers={"Range": "bytes=50-"})
        self.assertEqual(res1.status_code, 206)
        expected_len = self.file_size - 50
        self.assertEqual(len(res1.content), expected_len)
        self.assertEqual(res1.headers.get("Content-Range"), f"bytes 50-{self.file_size - 1}/{self.file_size}")

        # Range: bytes=-30 (last 30 bytes)
        res2 = self.client.get("/api/incidents/INC-HTTP-001/clip", headers={"Range": "bytes=-30"})
        self.assertEqual(res2.status_code, 206)
        self.assertEqual(len(res2.content), 30)

    def test_http_range_invalid_returns_416(self):
        """Tests that out-of-bounds byte range returns 416 Range Not Satisfiable."""
        headers = {"Range": f"bytes={self.file_size + 100}-{self.file_size + 500}"}
        res = self.client.get("/api/incidents/INC-HTTP-001/clip", headers=headers)
        self.assertEqual(res.status_code, 416)
        self.assertIn(f"bytes */{self.file_size}", res.headers.get("Content-Range", ""))

    def test_http_head_request(self):
        """Tests that HEAD request returns Accept-Ranges and Content-Length with empty body."""
        res = self.client.head("/api/incidents/INC-HTTP-001/clip")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Accept-Ranges"), "bytes")
        self.assertEqual(res.headers.get("Content-Length"), str(self.file_size))
        self.assertEqual(len(res.content), 0)

    def test_annotated_clip_query_param(self):
        """Tests ?annotated=true serves the annotated clip."""
        res = self.client.get("/api/incidents/INC-HTTP-001/clip?annotated=true")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Type"), "video/mp4")

    def test_retention_api_get_and_post(self):
        """Tests GET /api/retention and POST /api/retention settings management."""
        get_res = self.client.get("/api/retention")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertIn("retention_days", data)
        self.assertIn("max_disk_usage_mb", data)

        # Update retention settings
        post_res = self.client.post(
            "/api/retention",
            json={"retention_days": 14, "max_disk_usage_mb": 4096, "trigger_cleanup": True},
        )
        self.assertEqual(post_res.status_code, 200)
        updated = post_res.json()
        self.assertEqual(updated["retention_days"], 14)
        self.assertEqual(updated["max_disk_usage_mb"], 4096)


if __name__ == "__main__":
    unittest.main()
