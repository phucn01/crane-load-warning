"""Typed runtime and artifact contracts for the offline vision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

BBox = tuple[float, float, float, float]
BooleanMask = NDArray[np.bool_]


class Detection(TypedDict):
    """One normalised detector result in absolute image coordinates."""

    source_model: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    x1: float
    y1: float
    x2: float
    y2: float
    mask: BooleanMask | None


@dataclass(frozen=True, slots=True)
class DepthMapMetadata:
    """Description of a relative, explicitly non-metric depth map."""

    height: int
    width: int
    dtype: str
    finite_min: float
    finite_max: float
    finite_fraction: float
    convention: str = "relative_depth_not_metric"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelativeDepthResult:
    depth_map: NDArray[np.float32]
    metadata: DepthMapMetadata


@dataclass(frozen=True, slots=True)
class VisionFrameResult:
    """Deterministic vision output for one frame."""

    frame_id: str
    detections: tuple[Detection, ...]
    relative_depth: RelativeDepthResult


def clip_bbox(box: Any, image_shape: tuple[int, ...]) -> BBox | None:
    """Clip an xyxy box to an image and reject malformed/empty boxes."""

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        return None
    x1, y1, x2, y2 = values.tolist()
    x1, x2 = np.clip([x1, x2], 0.0, float(width)).tolist()
    y1, y2 = np.clip([y1, y2], 0.0, float(height)).tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def tensor_to_numpy(value: Any, *, columns: int | None = None) -> NDArray[Any]:
    """Move tensor-like data to CPU NumPy without requiring torch imports."""

    if value is None:
        shape = (0, columns) if columns is not None else (0,)
        return np.empty(shape, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if columns is not None:
        if array.size == 0:
            return np.empty((0, columns), dtype=np.float32)
        return array.reshape(-1, columns)
    return array


def detection_to_dict(
    detection: Detection,
    *,
    detection_id: str,
    mask_ref: str | None,
) -> dict[str, Any]:
    """Convert a runtime detection to JSON-safe artifact metadata."""

    x1, y1, x2, y2 = detection["bbox"]
    return {
        "detection_id": detection_id,
        "source_model": detection["source_model"],
        "class_id": int(detection["class_id"]),
        "class_name": detection["class_name"],
        "confidence": float(detection["confidence"]),
        "bbox": {
            "format": "xyxy_absolute",
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        },
        "has_mask": mask_ref is not None,
        "mask_ref": mask_ref,
    }


__all__ = [
    "BBox",
    "BooleanMask",
    "DepthMapMetadata",
    "Detection",
    "RelativeDepthResult",
    "VisionFrameResult",
    "clip_bbox",
    "detection_to_dict",
    "tensor_to_numpy",
]
