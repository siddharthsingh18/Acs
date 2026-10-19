"""
acs.temporal.features
Extracts physically grounded, camera-invariant temporal interaction features
from multi-object tracking history over 1 to 2 second sliding windows.

Features extracted per track pair:
1. Bounding-box overlap (IoU) trend (max, mean, slope, delta).
2. Closing speed (normalized distance rate of reduction, min distance, max closing speed).
3. Deceleration (rate of speed loss in norm units/sec^2, post-approach deceleration).
4. Direction change angle (sudden angular deflection in radians [0, pi]).
5. Time-to-collision (TTC) estimate (min TTC, mean TTC, critical fraction).
6. Post-overlap stationary time (duration in seconds vehicles remain stationary after contact).
7. Number of vehicles stopped (within track pair and across the scene).

All coordinates are normalized by (frame_width, frame_height) and temporal derivatives
by framerate (fps) so features transfer reliably across diverse camera viewpoints and FPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Stationary speed threshold in normalized frame units per second
# (e.g. 0.02 of frame width/height per second; at 1280x720 this is ~15-25 px/sec)
DEFAULT_STATIONARY_SPEED_THRESH: float = 0.025
SAFE_MAX_TTC: float = 10.0


@dataclass
class TrackPoint:
    """A single tracking observation at a given frame/time."""

    frame_idx: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # raw (x1, y1, x2, y2)
    norm_bbox: Tuple[float, float, float, float]  # normalized (x1/W, y1/H, x2/W, y2/H)
    centroid: Tuple[float, float]  # normalized (xc, yc)
    confidence: float
    class_id: int
    class_name: str
    track_id: int


def compute_box_iou_coords(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def angular_difference(theta1: float, theta2: float) -> float:
    """Computes shortest angular difference in radians [0, pi] between two headings."""
    diff = abs(theta1 - theta2) % (2.0 * math.pi)
    return diff if diff <= math.pi else (2.0 * math.pi - diff)


# Canonical ordered list of tabular feature names
FEATURE_NAMES: List[str] = [
    # 1. IoU trend
    "iou_max",
    "iou_mean",
    "iou_final",
    "iou_slope",
    "iou_delta",
    # 2. Closing speed
    "min_distance",
    "max_closing_speed",
    "mean_closing_speed",
    "final_distance",
    # 3. Deceleration
    "max_deceleration",
    "mean_deceleration",
    "post_approach_deceleration",
    "speed_drop_ratio",
    # 4. Direction change angle
    "max_direction_change_rad",
    "heading_convergence_rad",
    "heading_deflection_rad",
    # 5. Time-to-collision
    "min_ttc_seconds",
    "mean_ttc_seconds",
    "ttc_under_1s_ratio",
    # 6. Post-overlap stationary time
    "post_overlap_stationary_seconds",
    "post_overlap_stationary_ratio",
    "is_stationary_at_end",
    # 7. Stopped vehicles count
    "stopped_vehicles_in_pair",
    "stopped_vehicles_in_scene",
]

# Canonical sequence feature dimension (per time step)
SEQUENCE_FEATURE_NAMES: List[str] = [
    "iou",
    "distance",
    "closing_speed",
    "speed_a",
    "speed_b",
    "decel_a",
    "decel_b",
    "heading_diff",
    "ttc",
]


class TrackTrajectory:
    """Represents the temporal observation trajectory of a single track ID."""

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.points: List[TrackPoint] = []

    def add_point(self, point: TrackPoint) -> None:
        self.points.append(point)

    def prune_before(self, min_timestamp: float) -> None:
        """Removes observations older than min_timestamp."""
        self.points = [p for p in self.points if p.timestamp >= min_timestamp]

    def __len__(self) -> int:
        return len(self.points)

    def get_interpolated_points(self, fps: float = 30.0, max_gap_seconds: float = 2.0) -> Dict[int, TrackPoint]:
        """Returns dictionary of frame_idx -> TrackPoint with linear interpolation across gaps <= max_gap_seconds."""
        if not self.points:
            return {}

        sorted_pts = sorted(self.points, key=lambda p: p.frame_idx)
        interp_dict: Dict[int, TrackPoint] = {}

        for i in range(len(sorted_pts) - 1):
            p_start = sorted_pts[i]
            p_end = sorted_pts[i + 1]
            interp_dict[p_start.frame_idx] = p_start

            gap_frames = p_end.frame_idx - p_start.frame_idx
            gap_sec = p_end.timestamp - p_start.timestamp
            if 1 < gap_frames <= int(max_gap_seconds * fps + 0.5):
                # Interpolate intermediate frames
                for step in range(1, gap_frames):
                    alpha = step / float(gap_frames)
                    curr_f = p_start.frame_idx + step
                    curr_t = p_start.timestamp + alpha * gap_sec

                    # Interp norm_bbox and centroid
                    nb_start = p_start.norm_bbox
                    nb_end = p_end.norm_bbox
                    curr_nb = (
                        nb_start[0] + alpha * (nb_end[0] - nb_start[0]),
                        nb_start[1] + alpha * (nb_end[1] - nb_start[1]),
                        nb_start[2] + alpha * (nb_end[2] - nb_start[2]),
                        nb_start[3] + alpha * (nb_end[3] - nb_start[3]),
                    )
                    curr_cent = (
                        (curr_nb[0] + curr_nb[2]) / 2.0,
                        (curr_nb[1] + curr_nb[3]) / 2.0,
                    )
                    # Interp raw bbox
                    b_start = p_start.bbox
                    b_end = p_end.bbox
                    curr_b = (
                        b_start[0] + alpha * (b_end[0] - b_start[0]),
                        b_start[1] + alpha * (b_end[1] - b_start[1]),
                        b_start[2] + alpha * (b_end[2] - b_start[2]),
                        b_start[3] + alpha * (b_end[3] - b_start[3]),
                    )
                    interp_pt = TrackPoint(
                        frame_idx=curr_f,
                        timestamp=curr_t,
                        bbox=curr_b,
                        norm_bbox=curr_nb,
                        centroid=curr_cent,
                        confidence=p_start.confidence + alpha * (p_end.confidence - p_start.confidence),
                        class_id=p_start.class_id,
                        class_name=p_start.class_name,
                        track_id=self.track_id,
                    )
                    interp_dict[curr_f] = interp_pt

        interp_dict[sorted_pts[-1].frame_idx] = sorted_pts[-1]
        return interp_dict


class TrackPairFeatureExtractor:
    """Extracts normalized spatial-temporal interaction features for pairs of tracks."""

    def __init__(
        self,
        stationary_speed_thresh: float = DEFAULT_STATIONARY_SPEED_THRESH,
        fixed_sequence_length: int = 30,
    ):
        self.stationary_speed_thresh = stationary_speed_thresh
        self.fixed_sequence_length = fixed_sequence_length

    def extract_pair_features(
        self,
        traj_a: TrackTrajectory,
        traj_b: TrackTrajectory,
        window_seconds: float = 1.5,
        target_timestamp: Optional[float] = None,
        total_scene_stopped: int = 0,
        fps: float = 30.0,
    ) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Computes interaction features between two tracks over the specified time window.

        Args:
            traj_a: Trajectory of track A.
            traj_b: Trajectory of track B.
            window_seconds: Duration of window in seconds (e.g. 1.0 to 2.0s).
            target_timestamp: End time of window. Defaults to latest common timestamp.
            total_scene_stopped: Total stopped vehicle count in scene at window end.
            fps: Video framerate for velocity/derivative scaling.

        Returns:
            Tuple of (features_dict, tabular_vector, sequence_matrix), or None if insufficient history.
        """
        if len(traj_a) < 1 or len(traj_b) < 1:
            return None

        # Build interpolated mapping of timestamp / frame_idx
        pts_a = traj_a.get_interpolated_points(fps=fps)
        pts_b = traj_b.get_interpolated_points(fps=fps)

        common_frames = sorted(set(pts_a.keys()) & set(pts_b.keys()))
        if len(common_frames) < 2:
            return None

        # Filter to requested time window
        if target_timestamp is None:
            target_timestamp = max(pts_a[f].timestamp for f in common_frames)

        start_timestamp = target_timestamp - window_seconds
        window_frames = [
            f for f in common_frames
            if start_timestamp <= pts_a[f].timestamp <= target_timestamp + 0.05
        ]
        if len(window_frames) < 2:
            # Fall back to using the last available frames up to window duration
            window_frames = common_frames[-max(2, int(window_seconds * fps)):]

        if len(window_frames) < 2:
            return None

        # Compute per-frame quantities
        times: List[float] = []
        ious: List[float] = []
        distances: List[float] = []
        pos_a: List[Tuple[float, float]] = []
        pos_b: List[Tuple[float, float]] = []

        for f in window_frames:
            pa = pts_a[f]
            pb = pts_b[f]
            t = pa.timestamp
            times.append(t)
            pos_a.append(pa.centroid)
            pos_b.append(pb.centroid)

            iou = compute_box_iou_coords(pa.norm_bbox, pb.norm_bbox)
            ious.append(iou)

            dist = math.dist(pa.centroid, pb.centroid)
            distances.append(dist)

        n = len(window_frames)

        # Velocities and closing speeds
        speeds_a: List[float] = [0.0]
        speeds_b: List[float] = [0.0]
        headings_a: List[float] = [0.0]
        headings_b: List[float] = [0.0]
        closing_speeds: List[float] = [0.0]
        ttcs: List[float] = [SAFE_MAX_TTC]

        for i in range(1, n):
            dt = max(times[i] - times[i - 1], 1e-4)

            # Track A velocity
            dx_a = pos_a[i][0] - pos_a[i - 1][0]
            dy_a = pos_a[i][1] - pos_a[i - 1][1]
            v_a = math.hypot(dx_a, dy_a) / dt
            h_a = math.atan2(dy_a, dx_a)
            speeds_a.append(v_a)
            headings_a.append(h_a)

            # Track B velocity
            dx_b = pos_b[i][0] - pos_b[i - 1][0]
            dy_b = pos_b[i][1] - pos_b[i - 1][1]
            v_b = math.hypot(dx_b, dy_b) / dt
            h_b = math.atan2(dy_b, dx_b)
            speeds_b.append(v_b)
            headings_b.append(h_b)

            # Distance closing rate
            dist_change = distances[i - 1] - distances[i]
            v_close = dist_change / dt
            closing_speeds.append(v_close)

            # TTC
            if v_close > 1e-3 and distances[i] > 1e-4:
                ttc = min(SAFE_MAX_TTC, distances[i] / v_close)
            else:
                ttc = SAFE_MAX_TTC
            ttcs.append(ttc)

        # Decelerations (speed drop rate)
        decels_a: List[float] = [0.0]
        decels_b: List[float] = [0.0]
        for i in range(1, n):
            dt = max(times[i] - times[i - 1], 1e-4)
            a_a = (speeds_a[i] - speeds_a[i - 1]) / dt
            a_b = (speeds_b[i] - speeds_b[i - 1]) / dt
            decels_a.append(max(0.0, -a_a))
            decels_b.append(max(0.0, -a_b))

        # Direction changes
        dir_changes_a: List[float] = [0.0]
        dir_changes_b: List[float] = [0.0]
        for i in range(2, n):
            d_h_a = angular_difference(headings_a[i], headings_a[i - 1])
            d_h_b = angular_difference(headings_b[i], headings_b[i - 1])
            dir_changes_a.append(d_h_a)
            dir_changes_b.append(d_h_b)
        while len(dir_changes_a) < n:
            dir_changes_a.append(0.0)
            dir_changes_b.append(0.0)

        # 1. IoU trend features
        iou_max = float(np.max(ious))
        iou_mean = float(np.mean(ious))
        iou_final = float(ious[-1])
        iou_delta = float(ious[-1] - ious[0])

        # IoU linear slope
        t_arr = np.array(times) - times[0]
        if len(t_arr) > 2 and (t_arr[-1] - t_arr[0]) > 1e-4:
            iou_arr = np.array(ious)
            cov = np.cov(t_arr, iou_arr)[0, 1]
            var_t = np.var(t_arr)
            iou_slope = float(cov / var_t) if var_t > 1e-6 else 0.0
        else:
            iou_slope = 0.0

        # 2. Closing speed features
        min_distance = float(np.min(distances))
        max_closing_speed = float(np.max(closing_speeds))
        pos_closes = [cs for cs in closing_speeds if cs > 0]
        mean_closing_speed = float(np.mean(pos_closes)) if pos_closes else 0.0
        final_distance = float(distances[-1])

        # 3. Deceleration features
        joint_decels = [max(da, db) for da, db in zip(decels_a, decels_b)]
        max_deceleration = float(np.max(joint_decels))
        mean_deceleration = float(np.mean(joint_decels))

        min_dist_idx = int(np.argmin(distances))
        if min_dist_idx < n - 1:
            post_decels = joint_decels[min_dist_idx:]
            post_approach_deceleration = float(np.mean(post_decels))
        else:
            post_approach_deceleration = float(joint_decels[-1])

        # Speed drop ratio
        max_speed = max(max(speeds_a), max(speeds_b), 1e-4)
        end_speed = (speeds_a[-1] + speeds_b[-1]) / 2.0
        speed_drop_ratio = float(max(0.0, (max_speed - end_speed) / max_speed))

        # 4. Direction change angle features
        max_dir_change = max(max(dir_changes_a), max(dir_changes_b))
        max_direction_change_rad = float(max_dir_change)

        # Heading convergence (angle between vectors when approaching)
        if min_dist_idx > 0:
            conv_angle = angular_difference(headings_a[min_dist_idx], headings_b[min_dist_idx])
            heading_convergence_rad = float(conv_angle)
        else:
            heading_convergence_rad = float(angular_difference(headings_a[-1], headings_b[-1]))

        # Heading deflection (before vs after min distance)
        if min_dist_idx > 1 and min_dist_idx < n - 1:
            pre_h = math.atan2(
                pos_a[min_dist_idx][1] - pos_a[0][1],
                pos_a[min_dist_idx][0] - pos_a[0][0] + 1e-6,
            )
            post_h = math.atan2(
                pos_a[-1][1] - pos_a[min_dist_idx][1],
                pos_a[-1][0] - pos_a[min_dist_idx][0] + 1e-6,
            )
            heading_deflection_rad = float(angular_difference(pre_h, post_h))
        else:
            heading_deflection_rad = 0.0

        # 5. Time-to-collision features
        min_ttc_seconds = float(np.min(ttcs))
        mean_ttc_seconds = float(np.mean(ttcs))
        ttc_under_1s_ratio = float(np.mean([1.0 if t < 1.0 else 0.0 for t in ttcs]))

        # 6. Post-overlap stationary time features
        # Reference frame is max IoU or closest approach
        peak_idx = int(np.argmax(ious)) if iou_max > 0.05 else min_dist_idx
        stationary_time = 0.0
        stationary_frames = 0
        for i in range(peak_idx, n):
            is_stat_a = speeds_a[i] < self.stationary_speed_thresh
            is_stat_b = speeds_b[i] < self.stationary_speed_thresh
            if is_stat_a or is_stat_b:
                dt = times[i] - times[i - 1] if i > 0 else 1.0 / fps
                stationary_time += dt
                stationary_frames += 1

        total_post_time = max(times[-1] - times[peak_idx], 1e-4)
        post_overlap_stationary_seconds = float(stationary_time)
        post_overlap_stationary_ratio = float(min(1.0, stationary_time / total_post_time))
        is_stationary_at_end = float(
            1.0 if (speeds_a[-1] < self.stationary_speed_thresh or speeds_b[-1] < self.stationary_speed_thresh) else 0.0
        )

        # 7. Stopped vehicles count
        stopped_in_pair = 0
        if speeds_a[-1] < self.stationary_speed_thresh:
            stopped_in_pair += 1
        if speeds_b[-1] < self.stationary_speed_thresh:
            stopped_in_pair += 1
        stopped_vehicles_in_pair = float(stopped_in_pair)
        stopped_vehicles_in_scene = float(total_scene_stopped)

        feats: Dict[str, float] = {
            "iou_max": iou_max,
            "iou_mean": iou_mean,
            "iou_final": iou_final,
            "iou_slope": iou_slope,
            "iou_delta": iou_delta,
            "min_distance": min_distance,
            "max_closing_speed": max_closing_speed,
            "mean_closing_speed": mean_closing_speed,
            "final_distance": final_distance,
            "max_deceleration": max_deceleration,
            "mean_deceleration": mean_deceleration,
            "post_approach_deceleration": post_approach_deceleration,
            "speed_drop_ratio": speed_drop_ratio,
            "max_direction_change_rad": max_direction_change_rad,
            "heading_convergence_rad": heading_convergence_rad,
            "heading_deflection_rad": heading_deflection_rad,
            "min_ttc_seconds": min_ttc_seconds,
            "mean_ttc_seconds": mean_ttc_seconds,
            "ttc_under_1s_ratio": ttc_under_1s_ratio,
            "post_overlap_stationary_seconds": post_overlap_stationary_seconds,
            "post_overlap_stationary_ratio": post_overlap_stationary_ratio,
            "is_stationary_at_end": is_stationary_at_end,
            "stopped_vehicles_in_pair": stopped_vehicles_in_pair,
            "stopped_vehicles_in_scene": stopped_vehicles_in_scene,
        }

        # Tabular vector matching canonical FEATURE_NAMES
        tabular_vector = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)

        # Build sequence matrix [L, D] resampled / padded to fixed_sequence_length
        raw_seq_list = []
        for i in range(n):
            h_diff = angular_difference(headings_a[i], headings_b[i])
            step_feat = [
                ious[i],
                distances[i],
                closing_speeds[i],
                speeds_a[i],
                speeds_b[i],
                decels_a[i],
                decels_b[i],
                h_diff,
                ttcs[i],
            ]
            raw_seq_list.append(step_feat)

        raw_seq = np.array(raw_seq_list, dtype=np.float32)
        # Interpolate / resample to fixed_sequence_length
        target_len = self.fixed_sequence_length
        if len(raw_seq) == target_len:
            sequence_matrix = raw_seq
        else:
            x_old = np.linspace(0.0, 1.0, len(raw_seq))
            x_new = np.linspace(0.0, 1.0, target_len)
            seq_interp = np.zeros((target_len, raw_seq.shape[1]), dtype=np.float32)
            for d in range(raw_seq.shape[1]):
                seq_interp[:, d] = np.interp(x_new, x_old, raw_seq[:, d])
            sequence_matrix = seq_interp

        return feats, tabular_vector, sequence_matrix


class TrackHistoryBuffer:
    """Sliding time-buffer of multi-object detections for online temporal feature extraction."""

    def __init__(
        self,
        max_buffer_seconds: float = 3.5,
        stationary_speed_thresh: float = DEFAULT_STATIONARY_SPEED_THRESH,
        fps: float = 30.0,
    ):
        self.max_buffer_seconds = max_buffer_seconds
        self.stationary_speed_thresh = stationary_speed_thresh
        self.fps = fps
        self.trajectories: Dict[int, TrackTrajectory] = {}
        self.latest_timestamp: float = 0.0
        self._internal_track_counter: int = 0
        self.extractor = TrackPairFeatureExtractor(stationary_speed_thresh=stationary_speed_thresh)

    def add_frame_detections(
        self,
        frame_idx: int,
        timestamp: float,
        detections: Sequence[Any],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Appends all tracked entities from a frame into the history buffer."""
        self.latest_timestamp = max(self.latest_timestamp, timestamp)
        w = max(1.0, float(frame_width))
        h = max(1.0, float(frame_height))

        for det in detections:
            bbox = getattr(det, "bbox", (0.0, 0.0, 0.0, 0.0))
            norm_bbox = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
            centroid = ((norm_bbox[0] + norm_bbox[2]) / 2.0, (norm_bbox[1] + norm_bbox[3]) / 2.0)
            track_id = getattr(det, "track_id", None)
            if track_id is None:
                continue

            pt = TrackPoint(
                frame_idx=frame_idx,
                timestamp=timestamp,
                bbox=bbox,
                norm_bbox=norm_bbox,
                centroid=centroid,
                confidence=float(getattr(det, "confidence", 1.0)),
                class_id=int(getattr(det, "class_id", 0)),
                class_name=str(getattr(det, "class_name", "")),
                track_id=int(track_id),
            )

            if track_id not in self.trajectories:
                self.trajectories[track_id] = TrackTrajectory(track_id=int(track_id))
            self.trajectories[track_id].add_point(pt)

        # Prune points older than max_buffer_seconds
        cutoff = self.latest_timestamp - self.max_buffer_seconds
        for tid, traj in list(self.trajectories.items()):
            traj.prune_before(cutoff)
            if len(traj) == 0:
                del self.trajectories[tid]

    def count_scene_stopped(self, window_seconds: float = 1.0) -> int:
        """Counts how many active tracks in the scene are currently stationary."""
        stopped = 0
        for traj in self.trajectories.values():
            if len(traj) < 2:
                continue
            recent = [p for p in traj.points if p.timestamp >= self.latest_timestamp - window_seconds]
            if len(recent) >= 2:
                dt = max(recent[-1].timestamp - recent[0].timestamp, 1e-4)
                dist = math.dist(recent[-1].centroid, recent[0].centroid)
                speed = dist / dt
                if speed < self.stationary_speed_thresh:
                    stopped += 1
        return stopped

    def extract_candidate_pair_features(
        self,
        candidate_bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        window_seconds: float = 1.5,
    ) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray, Tuple[int, int]]]:
        """Finds the most critical track pair (e.g. interacting near candidate_bbox or closest approach)
        and extracts their temporal features.

        Returns:
            Tuple of (features_dict, tabular_vector, sequence_matrix, (track_a, track_b)), or None.
        """
        active_tids = list(self.trajectories.keys())
        if len(active_tids) < 2:
            return None

        total_stopped = self.count_scene_stopped(window_seconds=window_seconds)

        norm_cand_box = None
        if candidate_bbox is not None and frame_width and frame_height:
            w = max(1.0, float(frame_width))
            h = max(1.0, float(frame_height))
            norm_cand_box = (
                candidate_bbox[0] / w,
                candidate_bbox[1] / h,
                candidate_bbox[2] / w,
                candidate_bbox[3] / h,
            )

        best_result = None
        highest_risk = -1e9
        best_pair = (-1, -1)

        # Inspect all pairs
        for i in range(len(active_tids)):
            tid_a = active_tids[i]
            traj_a = self.trajectories[tid_a]
            for j in range(i + 1, len(active_tids)):
                tid_b = active_tids[j]
                traj_b = self.trajectories[tid_b]

                res = self.extractor.extract_pair_features(
                    traj_a=traj_a,
                    traj_b=traj_b,
                    window_seconds=window_seconds,
                    target_timestamp=self.latest_timestamp,
                    total_scene_stopped=total_stopped,
                    fps=self.fps,
                )
                if res is None:
                    continue

                feats, tab_vec, seq_mat = res

                # Risk score heuristic for selecting the primary collision candidate pair
                # Prioritize high IoU, high closing speed, sudden decel, or proximity to candidate box
                risk = (
                    feats["iou_max"] * 10.0
                    + feats["max_closing_speed"] * 3.0
                    + feats["max_deceleration"] * 2.0
                    + (5.0 - min(5.0, feats["min_ttc_seconds"]))
                    + feats["post_overlap_stationary_seconds"] * 2.0
                    + feats["stopped_vehicles_in_pair"] * 1.5
                )

                if norm_cand_box is not None:
                    # Boost if either track intersects candidate accident box
                    p_a = traj_a.points[-1].norm_bbox
                    p_b = traj_b.points[-1].norm_bbox
                    iou_a = compute_box_iou_coords(p_a, norm_cand_box)
                    iou_b = compute_box_iou_coords(p_b, norm_cand_box)
                    if max(iou_a, iou_b) > 0.05:
                        risk += 15.0

                if risk > highest_risk:
                    highest_risk = risk
                    best_result = (feats, tab_vec, seq_mat)
                    best_pair = (tid_a, tid_b)

        if best_result is not None:
            return best_result[0], best_result[1], best_result[2], best_pair

        return None

    def reset(self) -> None:
        """Clears all buffered trajectories."""
        self.trajectories.clear()
        self.latest_timestamp = 0.0
