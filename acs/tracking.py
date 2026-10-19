"""
acs.tracking
Enterprise MLflow experiment tracking integration for ACS.

Features:
- Links model versions, architectures, and checkpoints directly to datasets and metrics.
- Logs training hyperparameters, per-class mAP50, inference FPS benchmarks, and false alarms per hour.
- Registers promoted production models into the MLflow Model Registry with versioning.
- Gracefully handles offline or unconfigured environments without crashing training or evaluation pipelines.
"""
import hashlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

logger = logging.getLogger("acs.tracking")

DEFAULT_EXPERIMENT_NAME = "acs-accident-detection"
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns.db")
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_DB_PATH}")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Computes a SHA256 checksum of a file to fingerprint dataset configs and model weights."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    if not path.is_file():
        return "missing"
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except Exception as e:
        logger.debug("Failed hashing file %s: %s", path, e)
        return "hash_error"


class ExperimentTracker:
    """Wrapper around MLflow tracking API with safety fallbacks and dataset/model linking."""

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("MLFLOW_DISABLED", "false").lower() not in ("true", "1", "yes")
        )
        self._mlflow = None
        self._active_run = None

        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        """Initializes MLflow client and verifies local/remote experiment configuration."""
        try:
            import mlflow
            self._mlflow = mlflow
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            logger.info(
                "MLflow tracking initialized [Experiment: '%s' | URI: %s]",
                self.experiment_name,
                self.tracking_uri,
            )
        except ImportError:
            logger.warning("mlflow package not installed; experiment tracking disabled.")
            self.enabled = False
        except Exception as e:
            logger.warning("Failed initializing MLflow tracking (%s); running in fallback mode.", e)
            self.enabled = False

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ) -> Generator[Optional[Any], None, None]:
        """Context manager for an MLflow experiment run."""
        if not self.enabled or not self._mlflow:
            yield None
            return

        try:
            with self._mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
                self._active_run = run
                yield run
        except Exception as e:
            logger.warning("MLflow start_run error: %s", e)
            yield None
        finally:
            self._active_run = None

    def log_dataset_metadata(
        self,
        name: str,
        data_yaml_path: Optional[Union[str, Path]] = None,
        num_samples: Optional[Dict[str, int]] = None,
        version: Optional[str] = None,
    ) -> None:
        """Logs dataset fingerprint, yaml config path, and split sample counts linked to the run."""
        if not self.enabled or not self._mlflow:
            return

        params = {"dataset_name": name}
        if version:
            params["dataset_version"] = version

        if data_yaml_path:
            yaml_path = Path(data_yaml_path)
            params["dataset_config"] = str(yaml_path.name)
            if yaml_path.is_file():
                params["dataset_config_hash"] = compute_file_hash(yaml_path)
                try:
                    self._mlflow.log_artifact(str(yaml_path), artifact_path="dataset_config")
                except Exception:
                    pass

        if num_samples:
            for split, count in num_samples.items():
                params[f"dataset_{split}_count"] = count

        try:
            self._mlflow.log_params(params)
        except Exception as e:
            logger.debug("Failed logging dataset metadata to MLflow: %s", e)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs training or evaluation hyperparameters."""
        if not self.enabled or not self._mlflow:
            return
        # Flatten non-primitive values to strings
        cleaned = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                cleaned[k] = v
            elif v is None:
                cleaned[k] = "None"
            else:
                cleaned[k] = str(v)[:250]
        try:
            self._mlflow.log_params(cleaned)
        except Exception as e:
            logger.debug("Failed logging params to MLflow: %s", e)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Logs quantitative evaluation metrics (e.g. mAP50, precision, recall, CPU FPS, FAR)."""
        if not self.enabled or not self._mlflow:
            return
        cleaned = {}
        for k, v in metrics.items():
            try:
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):  # check not NaN
                    cleaned[k] = float(v)
            except Exception:
                pass
        try:
            self._mlflow.log_metrics(cleaned, step=step)
        except Exception as e:
            logger.debug("Failed logging metrics to MLflow: %s", e)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Uploads a report, plot, or log file to MLflow artifacts."""
        if not self.enabled or not self._mlflow:
            return
        p = Path(local_path)
        if not p.exists():
            return
        try:
            self._mlflow.log_artifact(str(p), artifact_path=artifact_path)
        except Exception as e:
            logger.debug("Failed logging artifact %s to MLflow: %s", p, e)

    def log_and_register_model(
        self,
        model_path: Union[str, Path],
        registered_model_name: Optional[str] = "acs-yolo-accident-detector",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Logs model checkpoint file and registers it in the Model Registry if configured."""
        if not self.enabled or not self._mlflow:
            return
        path = Path(model_path)
        if not path.is_file():
            return

        model_hash = compute_file_hash(path)
        self.log_params({
            "model_filename": path.name,
            "model_size_bytes": path.stat().st_size,
            "model_sha256": model_hash,
        })
        self.log_artifact(str(path), artifact_path="model_weights")

        if registered_model_name:
            try:
                model_uri = f"runs:/{self._active_run.info.run_id}/model_weights/{path.name}" if self._active_run else None
                if model_uri:
                    self._mlflow.register_model(
                        model_uri=model_uri,
                        name=registered_model_name,
                        tags=tags or {"framework": "ultralytics_yolo", "sha256": model_hash},
                    )
                    logger.info("Registered model '%s' in MLflow Registry.", registered_model_name)
            except Exception as e:
                logger.debug("Model registry registration notice: %s", e)


# Global singleton tracker
tracker = ExperimentTracker()
