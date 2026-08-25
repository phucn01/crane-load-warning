"""Load and cache Phase-1 vision models exactly once per process/run."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from .depth import DepthAnythingV3, DepthAnythingV3Config
from .detectors import (
    RFDETRLoadDetector,
    RFDETRLoadDetectorConfig,
    YOLOPersonSegmenter,
    YOLOPersonSegmenterConfig,
)

ModelFactory = Callable[[], Any]
WarmupCallback = Callable[[Any], None]
LOGGER = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe registry that creates, loads, and reuses model adapters."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}
        self._warmups: dict[str, WarmupCallback | None] = {}
        self._instances: dict[str, Any] = {}
        self._lock = Lock()

    def register(
        self,
        name: str,
        factory: ModelFactory,
        *,
        warmup: WarmupCallback | None = None,
    ) -> None:
        if not name:
            raise ValueError("model name must not be empty")
        with self._lock:
            if name in self._factories or name in self._instances:
                raise ValueError(f"model is already registered: {name}")
            self._factories[name] = factory
            self._warmups[name] = warmup

    def get(self, name: str) -> Any:
        with self._lock:
            if name in self._instances:
                LOGGER.debug("model_cache_hit model=%s", name)
                return self._instances[name]
            if name not in self._factories:
                raise KeyError(f"model is not registered: {name}")

            LOGGER.info("=== START | COMPONENT=VISION | OPERATION=MODEL_LOAD | MODEL=%s ===", name)
            started = perf_counter()
            try:
                instance = self._factories[name]()
                if hasattr(instance, "load"):
                    instance.load()
                warmup = self._warmups[name]
                if warmup is not None:
                    warmup(instance)
                self._instances[name] = instance
            except BaseException as error:
                LOGGER.exception(
                    "=== ERROR | COMPONENT=VISION | OPERATION=MODEL_LOAD | "
                    "MODEL=%s | DURATION_MS=%.3f | ERROR_TYPE=%s ===",
                    name,
                    (perf_counter() - started) * 1000.0,
                    type(error).__name__,
                )
                raise
            LOGGER.info(
                "=== END | COMPONENT=VISION | OPERATION=MODEL_LOAD | MODEL=%s | "
                "DURATION_MS=%.3f ===",
                name,
                (perf_counter() - started) * 1000.0,
            )
            return instance

    def load_all(self) -> None:
        for name in tuple(self._factories):
            self.get(name)

    def metadata(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for name in self._factories:
            instance = self._instances.get(name)
            if instance is None:
                models[name] = {"loaded": False}
            elif hasattr(instance, "metadata"):
                models[name] = instance.metadata()
            else:
                models[name] = {"loaded": True}
        return {
            "models": models,
            "packages": {
                package: _package_version(package)
                for package in (
                    "rfdetr",
                    "ultralytics",
                    "depth-anything-3",
                    "torch",
                    "numpy",
                    "opencv-python",
                )
            },
        }


def build_model_manager(
    config: Mapping[str, Any],
    *,
    config_dir: str | Path,
) -> ModelManager:
    """Build the three Phase-1 adapters from parsed models YAML."""

    base_dir = Path(config_dir).resolve()
    default_device = _normalise_device(config.get("device"))
    rfdetr = _section(config, "rfdetr")
    yolo = _section(config, "yolo_person")
    da3 = _section(config, "depth_anything_v3")

    rfdetr_checkpoint = _resolve_required_path(
        rfdetr.get("checkpoint"), base_dir, "rfdetr.checkpoint"
    )
    yolo_checkpoint = _resolve_required_path(
        yolo.get("checkpoint"), base_dir, "yolo_person.checkpoint"
    )

    class_names_raw = rfdetr.get(
        "class_names", {0: "hanging_object", 1: "hanging_rope"}
    )
    if not isinstance(class_names_raw, Mapping):
        raise TypeError("rfdetr.class_names must be a mapping")
    class_names = {int(key): str(value) for key, value in class_names_raw.items()}

    manager = ModelManager()
    manager.register(
        "rfdetr",
        lambda: RFDETRLoadDetector(
            rfdetr_checkpoint,
            config=RFDETRLoadDetectorConfig(
                confidence=float(rfdetr.get("confidence", 0.25)),
                device=_normalise_device(rfdetr.get("device", default_device)),
                num_classes=int(rfdetr.get("num_classes", 2)),
                class_names=class_names,
            ),
        ),
    )
    manager.register(
        "yolo_person",
        lambda: YOLOPersonSegmenter(
            yolo_checkpoint,
            config=YOLOPersonSegmenterConfig(
                confidence=float(yolo.get("confidence", 0.35)),
                image_size=int(yolo.get("image_size", 640)),
                device=_normalise_device(yolo.get("device", default_device)),
                person_class_id=int(yolo.get("person_class_id", 0)),
                retina_masks=bool(yolo.get("retina_masks", True)),
                auto_download=bool(yolo.get("auto_download", True)),
            ),
        ),
    )
    manager.register(
        "depth_anything_v3",
        lambda: DepthAnythingV3(
            config=DepthAnythingV3Config(
                model_name=str(da3.get("model_name", "depth-anything/DA3-BASE")),
                device=_normalise_device(da3.get("device", default_device)),
                process_resolution=int(da3.get("process_resolution", 504)),
                local_files_only=bool(da3.get("local_files_only", False)),
            )
        ),
    )
    return manager


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _resolve_required_path(value: Any, base_dir: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _normalise_device(value: Any) -> str | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return f"cuda:{value}"
    device = str(value).strip().lower()
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


__all__ = ["ModelManager", "build_model_manager"]
