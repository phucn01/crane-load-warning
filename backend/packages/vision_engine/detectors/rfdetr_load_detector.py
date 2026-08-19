"""RF-DETR adapter for hanging-object and hanging-rope detection."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..contracts import Detection, clip_bbox, tensor_to_numpy


DEFAULT_CLASS_NAMES = {0: "hanging_object", 1: "hanging_rope"}


@dataclass(frozen=True, slots=True)
class RFDETRLoadDetectorConfig:
    confidence: float = 0.25
    device: str | int | None = None
    num_classes: int = 2
    class_names: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_NAMES))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than zero")


class RFDETRLoadDetector:
    """Normalise RF-DETR predictions into the shared detection contract."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        config: RFDETRLoadDetectorConfig | None = None,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if model is None and checkpoint_path is None:
            raise ValueError("checkpoint_path is required when model is not provided")
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.config = config or RFDETRLoadDetectorConfig()
        self._model = model
        self._model_factory = model_factory
        self._lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self.checkpoint_path is None:
            raise RuntimeError("RF-DETR checkpoint path is not configured")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RF-DETR checkpoint not found: {self.checkpoint_path}"
            )

        factory = self._model_factory
        if factory is None:
            try:
                from rfdetr.variants import RFDETRMedium
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("rfdetr is required to load RF-DETR Medium") from exc
            factory = RFDETRMedium

        kwargs = {"pretrain_weights": str(self.checkpoint_path)}
        try:
            model = factory(num_classes=self.config.num_classes, **kwargs)
        except TypeError:
            # Support RF-DETR versions without num_classes.
            model = factory(**kwargs)
        if self.config.device is not None and hasattr(model, "to"):
            device = (
                f"cuda:{self.config.device}"
                if isinstance(self.config.device, int)
                else self.config.device
            )
            model.to(device)
        self._model = model
        return model

    def load(self) -> "RFDETRLoadDetector":
        with self._lock:
            self._get_model()
        return self

    def predict(self, image_bgr: NDArray[np.generic]) -> list[Detection]:
        _validate_image(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            _install_supervision_compat_if_needed()
            prediction = self._get_model().predict(
                Image.fromarray(image_rgb), threshold=self.config.confidence
            )

        boxes = tensor_to_numpy(
            getattr(prediction, "xyxy", None), columns=4
        )
        scores = tensor_to_numpy(
            getattr(prediction, "confidence", None)
        ).reshape(-1)
        classes = tensor_to_numpy(
            getattr(prediction, "class_id", None)
        ).reshape(-1)

        detections: list[Detection] = []
        for index, box in enumerate(boxes):
            bbox = clip_bbox(box, image_bgr.shape)
            if bbox is None:
                continue
            class_id = int(classes[index]) if index < len(classes) else -1
            confidence = float(scores[index]) if index < len(scores) else 0.0
            x1, y1, x2, y2 = bbox
            detections.append(
                {
                    "source_model": "RF-DETR Medium",
                    "class_id": class_id,
                    "class_name": self.config.class_names.get(
                        class_id, f"class_{class_id}"
                    ),
                    "confidence": confidence,
                    "bbox": bbox,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "mask": None,
                }
            )
        return detections

    __call__ = predict

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "RF-DETR Medium",
            "provider": "rfdetr",
            "identifier": (
                str(self.checkpoint_path)
                if self.checkpoint_path is not None
                else "injected_model"
            ),
            "device": self.config.device,
            "confidence_threshold": self.config.confidence,
            "class_names": self.config.class_names,
            "loaded": self._model is not None,
        }


def _validate_image(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RF-DETR image must have shape (height, width, 3)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be greater than zero")


def _install_supervision_compat_if_needed() -> None:
    """Use minimal RF-DETR result types if supervision cannot import."""

    if "supervision" in sys.modules:
        return
    try:
        __import__("supervision")
        return
    except ImportError:
        pass

    class Detections:
        def __init__(self, **kwargs: Any) -> None:
            self.xyxy = kwargs.get("xyxy")
            self.confidence = kwargs.get("confidence")
            self.class_id = kwargs.get("class_id")
            self.mask = kwargs.get("mask")
            self.data: dict[str, Any] = {}
            self.metadata: dict[str, Any] = {}

        def __len__(self) -> int:
            return 0 if self.xyxy is None else len(self.xyxy)

    module = types.ModuleType("supervision")
    module.Detections = Detections
    module.KeyPoints = type("KeyPoints", (), {})
    sys.modules["supervision"] = module


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "RFDETRLoadDetector",
    "RFDETRLoadDetectorConfig",
]
