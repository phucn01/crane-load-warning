"""YOLO26m adapter for person segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from ..contracts import (
    BBox,
    BooleanMask,
    Detection,
    clip_bbox,
    tensor_to_numpy,
)


PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

PersonSegmentation = Detection


@dataclass(frozen=True, slots=True)
class YOLOPersonSegmenterConfig:
    """YOLO inference settings."""

    confidence: float = 0.35
    image_size: int = 640
    device: str | int | None = None
    person_class_id: int = PERSON_CLASS_ID
    retina_masks: bool = True
    auto_download: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if self.person_class_id < 0:
            raise ValueError("person_class_id must be non-negative")


class YOLOPersonSegmenter:
    """Run YOLO26m segmentation and expose a stable person result contract.

    The Ultralytics model is loaded lazily so importing the vision package does
    not allocate model/GPU resources.  A pre-built ``model`` can be injected in
    tests or by an application-level model registry.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        config: YOLOPersonSegmenterConfig | None = None,
        model: Any | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError("model_path is required when model is not provided")

        self.model_path = Path(model_path) if model_path is not None else None
        self.config = config or YOLOPersonSegmenterConfig()
        self._model = model
        self._model_factory = model_factory
        self._lock = Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self.model_path is None:
            raise RuntimeError("YOLO model path is not configured")
        if not self.model_path.is_file() and not self.config.auto_download:
            raise FileNotFoundError(f"YOLO model file not found: {self.model_path}")

        factory = self._model_factory
        if factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "ultralytics is required to load the YOLO26m model"
                ) from exc
            factory = YOLO

        # Create the target folder before Ultralytics downloads the model.
        if self.config.auto_download and not self.model_path.is_file():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = factory(str(self.model_path))
        return self._model

    def load(self) -> "YOLOPersonSegmenter":
        """Load model weights now instead of waiting for first prediction."""

        with self._lock:
            self._get_model()
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "YOLO26m segmentation",
            "provider": "ultralytics",
            "identifier": str(self.model_path) if self.model_path else "injected_model",
            "device": self.config.device,
            "confidence_threshold": self.config.confidence,
            "image_size": self.config.image_size,
            "auto_download": self.config.auto_download,
            "local_file_available": bool(
                self.model_path is not None and self.model_path.is_file()
            ),
            "loaded": self._model is not None,
        }

    def predict(self, image: NDArray[np.generic]) -> list[PersonSegmentation]:
        """Detect people in a BGR or RGB image.

        The ndarray is forwarded unchanged and this adapter does not mutate the
        caller's image.  Its channel convention must therefore match the one
        expected by the configured Ultralytics model.  Results are flattened
        if Ultralytics returns more than one result object.
        """

        _validate_image(image)
        height, width = image.shape[:2]

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "classes": [self.config.person_class_id],
            "conf": self.config.confidence,
            "imgsz": self.config.image_size,
            "retina_masks": self.config.retina_masks,
            "verbose": False,
        }
        if self.config.device is not None:
            predict_kwargs["device"] = self.config.device

        # Protect the shared Ultralytics predictor state.
        with self._lock:
            results = self._get_model().predict(**predict_kwargs)

        detections: list[PersonSegmentation] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = tensor_to_numpy(getattr(boxes, "xyxy", None), columns=4)
            scores = tensor_to_numpy(getattr(boxes, "conf", None)).reshape(-1)
            classes = tensor_to_numpy(getattr(boxes, "cls", None)).reshape(-1)
            masks = _normalise_masks(getattr(result, "masks", None), (height, width))

            # Boxes and masks use matching indexes.
            for index, box in enumerate(xyxy):
                class_id = int(classes[index]) if index < len(classes) else -1
                if class_id != self.config.person_class_id:
                    continue

                bbox = clip_bbox(box, (height, width))
                if bbox is None:
                    continue

                confidence = float(scores[index]) if index < len(scores) else 0.0
                mask = masks[index] if index < len(masks) else None
                x1, y1, x2, y2 = bbox
                detections.append(
                    {
                        "source_model": "YOLO26m",
                        "class_id": class_id,
                        "class_name": PERSON_CLASS_NAME,
                        "confidence": confidence,
                        "bbox": bbox,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "mask": mask,
                    }
                )

        return detections

    __call__ = predict


def run_person_detection(
    image: NDArray[np.generic],
    *,
    model_path: str | Path | None = None,
    config: YOLOPersonSegmenterConfig | None = None,
    model: Any | None = None,
) -> list[PersonSegmentation]:
    """Run person segmentation without managing an adapter instance.

    Long-running services should construct one :class:`YOLOPersonSegmenter`
    and reuse it rather than calling this helper for every frame.
    """

    return YOLOPersonSegmenter(
        model_path=model_path,
        config=config,
        model=model,
    ).predict(image)


def _validate_image(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (height, width) or (height, width, channels)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be greater than zero")


def _normalise_masks(masks: Any, image_shape: tuple[int, int]) -> list[BooleanMask | None]:
    if masks is None:
        return []

    polygons = getattr(masks, "xy", None)
    if polygons is not None:
        return [_polygon_to_mask(polygon, image_shape) for polygon in polygons]

    data = getattr(masks, "data", None)
    if data is None:
        return []
    mask_data = tensor_to_numpy(data)
    if mask_data.ndim == 2:
        mask_data = mask_data[np.newaxis, ...]
    if mask_data.ndim != 3:
        return []
    return [_raster_to_mask(mask, image_shape) for mask in mask_data]


def _polygon_to_mask(polygon: Any, image_shape: tuple[int, int]) -> BooleanMask | None:
    mask = np.zeros(image_shape, dtype=np.uint8)
    contours: list[NDArray[np.int32]] = []

    # Support one or multiple contours per detection.
    candidates = polygon if isinstance(polygon, (list, tuple)) else [polygon]
    for candidate in candidates:
        points = np.asarray(candidate, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            contours.append(np.rint(points).astype(np.int32))

    if not contours:
        return None
    cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool) if np.any(mask) else None


def _raster_to_mask(mask: Any, image_shape: tuple[int, int]) -> BooleanMask | None:
    array = np.asarray(mask, dtype=np.float32)
    if array.shape != image_shape:
        array = cv2.resize(
            array,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    boolean_mask = array > 0.5
    return boolean_mask if np.any(boolean_mask) else None


__all__ = [
    "BBox",
    "BooleanMask",
    "PERSON_CLASS_ID",
    "PERSON_CLASS_NAME",
    "PersonSegmentation",
    "YOLOPersonSegmenter",
    "YOLOPersonSegmenterConfig",
    "run_person_detection",
]
