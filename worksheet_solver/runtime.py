"""Shared, thread-safe runtime for the worksheet detection model."""

import os
from pathlib import Path
from threading import Lock
from typing import Any

from ultralytics import YOLO


class DetectionModelRuntime:
    """Own one YOLO model and serialize access to its inference method."""

    def __init__(self, model_path: str | Path):
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Gap detection model not found: {resolved_path}"
            )

        self.model_path = resolved_path
        self.model = YOLO(str(resolved_path))
        self.inference_lock = Lock()

    def predict(self, **kwargs: Any):
        """Run one inference while excluding concurrent access to YOLO."""
        with self.inference_lock:
            return self.model.predict(**kwargs)


_RUNTIME_CACHE: dict[str, DetectionModelRuntime] = {}
_RUNTIME_CACHE_LOCK = Lock()


def get_detection_model_runtime(
    model_path: str | Path,
) -> DetectionModelRuntime:
    """Return the process-wide runtime for a detector weights file."""
    resolved_path = Path(model_path).expanduser().resolve()
    cache_key = os.path.normcase(str(resolved_path))
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(cache_key)
        if runtime is None:
            runtime = DetectionModelRuntime(resolved_path)
            _RUNTIME_CACHE[cache_key] = runtime
        return runtime


def clear_detection_model_cache() -> None:
    """Clear cached runtimes; intended for controlled shutdowns and tests."""
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE.clear()
