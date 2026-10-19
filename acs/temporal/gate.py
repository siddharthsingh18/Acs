"""
acs.temporal.gate
High-performance second-stage temporal model gate for confirming or suppressing
single-frame YOLO accident proposals using 1 to 2 second track histories.

Supports:
- PyTorch sequence models (.pt, e.g. 1D-CNN or GRU)
- LightGBM / Scikit-Learn tabular models (.joblib)
- Fallback heuristics when tracking history is sparse or unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger("acs.temporal.gate")


class TemporalGate:
    """Second-stage temporal verification gate that confirms or suppresses
    YOLO accident proposals using multi-object kinematic track interactions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.50,
        device: str = "cpu",
        enabled: bool = True,
    ):
        self.model_path = model_path or "models/temporal_gate.pt"
        self.threshold = threshold
        self.device = device
        self.enabled = enabled
        self.model = None
        self.model_type: str = "disabled"

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Loads either a PyTorch sequence checkpoint (.pt) or a LightGBM checkpoint (.joblib)."""
        path = Path(self.model_path)
        if not path.is_file():
            # Try relative to repo root
            root_path = Path(__file__).resolve().parent.parent.parent / self.model_path
            if root_path.is_file():
                path = root_path
            else:
                logger.warning(
                    "Temporal gate model file not found at '%s'. Temporal gate will operate in pass-through mode.",
                    self.model_path,
                )
                self.model = None
                return

        try:
            if path.suffix.lower() == ".pt":
                import torch
                from ml.scripts.train_temporal_model import Temporal1DCNN, TemporalGRU

                # Detect architecture from state_dict keys
                state_dict = torch.load(str(path), map_location=self.device)
                if "conv1.weight" in state_dict:
                    self.model = Temporal1DCNN(in_channels=9, hidden_dim=32)
                    self.model_type = "1d_cnn"
                else:
                    self.model = TemporalGRU(in_dim=9, hidden_dim=32)
                    self.model_type = "gru"

                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Loaded PyTorch temporal gate (%s) from %s", self.model_type, path)

            elif path.suffix.lower() in (".joblib", ".pkl"):
                import joblib
                self.model = joblib.load(str(path))
                self.model_type = "lightgbm"
                logger.info("Loaded LightGBM temporal gate from %s", path)
            else:
                logger.warning("Unrecognized temporal model format: %s", path.suffix)
                self.model = None

        except Exception as exc:
            logger.error("Failed loading temporal gate model from %s: %s", path, exc)
            self.model = None

    def verify(
        self,
        history_buffer: Any,
        candidate_bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        window_seconds: float = 1.5,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Evaluates recent track interactions to confirm or suppress an accident proposal.

        Args:
            history_buffer: TrackHistoryBuffer holding recent object tracks.
            candidate_bbox: Raw bounding box (x1, y1, x2, y2) of YOLO proposal.
            frame_width: Current frame width.
            frame_height: Current frame height.
            window_seconds: Evaluation time window (1.0 to 2.0s).

        Returns:
            Tuple of (is_confirmed: bool, probability: float, diagnostics: Dict[str, Any]).
        """
        if not self.enabled or self.model is None:
            # Pass-through if disabled or unloaded
            return True, 1.0, {"gate_status": "disabled_or_unloaded"}

        if not getattr(history_buffer, "trajectories", None):
            # Pass-through if tracking is not running or mock test lacks trajectories
            return True, 1.0, {"gate_status": "empty_history_buffer"}

        # Extract interacting track pair features
        pair_result = history_buffer.extract_candidate_pair_features(
            candidate_bbox=candidate_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            window_seconds=window_seconds,
        )

        if pair_result is None:
            # When tracking history is sparse (e.g. no interacting vehicle pair found):
            # Check if any single vehicle is stopped in the scene
            stopped_scene = history_buffer.count_scene_stopped(window_seconds=window_seconds)
            if stopped_scene >= 1:
                # Scene has stopped vehicle(s) consistent with crash
                prob = 0.55
                diagnostics = {"sparse_tracking": True, "stopped_scene": stopped_scene}
                return prob >= self.threshold, prob, diagnostics
            else:
                # High speed movement or no tracking; single-frame proposal is likely false alarm
                prob = 0.15
                diagnostics = {"sparse_tracking": True, "stopped_scene": 0}
                return prob >= self.threshold, prob, diagnostics

        feats, tab_vec, seq_mat, track_pair = pair_result

        # Inference
        try:
            if self.model_type == "lightgbm":
                # LightGBM predict_proba
                prob = float(self.model.predict_proba(tab_vec.reshape(1, -1))[0, 1])
            else:
                # PyTorch sequence model
                import torch
                with torch.no_grad():
                    seq_tensor = torch.tensor(seq_mat, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logit = self.model(seq_tensor)
                    prob = float(torch.sigmoid(logit).item())

            is_confirmed = prob >= self.threshold
            diagnostics = {
                "probability": round(prob, 4),
                "threshold": self.threshold,
                "is_confirmed": is_confirmed,
                "model_type": self.model_type,
                "track_pair": list(track_pair),
                "top_metrics": {
                    "iou_max": round(feats.get("iou_max", 0.0), 3),
                    "min_distance": round(feats.get("min_distance", 0.0), 3),
                    "max_deceleration": round(feats.get("max_deceleration", 0.0), 3),
                    "post_overlap_stationary_seconds": round(feats.get("post_overlap_stationary_seconds", 0.0), 3),
                    "stopped_vehicles_in_pair": int(feats.get("stopped_vehicles_in_pair", 0)),
                },
            }
            return is_confirmed, prob, diagnostics

        except Exception as exc:
            logger.warning("Temporal gate inference error: %s", exc)
            return True, 0.50, {"inference_error": str(exc)}
