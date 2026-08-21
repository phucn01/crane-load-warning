"""Representative image-point and relative-depth selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vision_engine.contracts import Detection, clip_bbox

from .config import RepresentativeDepthConfig
from .contracts import (
    DepthQuality,
    DepthStatistics,
    ImagePoint,
    PersonRepresentative,
    RepresentativeDepth,
)

PERSON_TOP_FRACTION = 0.60
PERSON_LOWER_FRACTION = 0.40
PERSON_FOOT_FRACTION = 0.20
HIGH_QUALITY_DIFFERENCE = 0.10
MEDIUM_QUALITY_DIFFERENCE = 0.25


def estimate_person_representative(
    detection: Detection,
    depth_map: NDArray[np.generic],
    *,
    config: RepresentativeDepthConfig | None = None,
) -> PersonRepresentative:
    """Select one person point and one non-metric representative depth."""

    settings = config or RepresentativeDepthConfig()
    _validate_detection_class(detection, "person")
    _validate_depth_map(depth_map)

    representative_point, representative_point_source = person_representative_point(
        detection,
        image_shape=depth_map.shape,
        bottom_fraction=settings.person_bottom_fraction,
    )
    representative_depth = person_representative_depth(
        detection,
        depth_map,
        minimum_valid_pixels=settings.minimum_valid_pixels,
    )
    return PersonRepresentative(
        point=representative_point,
        point_source=representative_point_source,
        depth=representative_depth,
    )


def person_representative_point(
    detection: Detection,
    *,
    image_shape: tuple[int, ...],
    bottom_fraction: float = 0.10,
) -> tuple[ImagePoint, str]:
    """Use the median bottom mask band, then bbox bottom-center fallback."""

    _validate_detection_class(detection, "person")
    if not 0.0 < bottom_fraction <= 1.0:
        raise ValueError("bottom_fraction must be in the range (0, 1]")

    mask = _valid_detection_mask(detection, image_shape)
    if mask is not None:
        ys, xs = np.nonzero(mask)
        bottom_band_y_cutoff = np.percentile(
            ys,
            100.0 * (1.0 - bottom_fraction),
        )
        bottom_band_mask = ys >= bottom_band_y_cutoff
        if np.any(bottom_band_mask):
            return (
                ImagePoint(
                    x=float(np.median(xs[bottom_band_mask])),
                    y=float(np.median(ys[bottom_band_mask])),
                ),
                "segmentation_bottom_band",
            )

        # Defensive fallback; a non-empty mask normally always has a bottom band.
        return (
            ImagePoint(x=float(np.median(xs)), y=float(np.median(ys))),
            "segmentation_median",
        )

    bbox = clip_bbox(detection["bbox"], image_shape)
    if bbox is None:
        raise ValueError("person detection has no valid bbox or mask")
    x1, _, x2, y2 = bbox
    return ImagePoint(x=(x1 + x2) / 2.0, y=y2), "bbox_bottom_center"


def person_representative_depth(
    detection: Detection,
    depth_map: NDArray[np.generic],
    *,
    minimum_valid_pixels: int = 5,
) -> RepresentativeDepth:
    """Estimate person depth using mask-first full/top/lower/foot ROIs."""

    _validate_detection_class(detection, "person")
    _validate_depth_map(depth_map)
    _validate_minimum_valid_pixels(minimum_valid_pixels)

    bbox = _required_bbox(detection, depth_map.shape)
    mask = _valid_detection_mask(detection, depth_map.shape)
    base_source = "segmentation_mask" if mask is not None else "bbox_fallback"
    detection_region_mask = (
        mask if mask is not None else _bbox_region(depth_map.shape, bbox)
    )

    roi_masks = {
        "full": detection_region_mask,
        "top": detection_region_mask
        & _bbox_region(depth_map.shape, _top_bbox(bbox, PERSON_TOP_FRACTION)),
        "lower": detection_region_mask
        & _bbox_region(depth_map.shape, _bottom_bbox(bbox, PERSON_LOWER_FRACTION)),
        "foot": detection_region_mask
        & _bbox_region(depth_map.shape, _bottom_bbox(bbox, PERSON_FOOT_FRACTION)),
    }
    roi_statistics = {
        name: calculate_depth_statistics(
            depth_map[region],
            minimum_valid_pixels=minimum_valid_pixels,
        )
        for name, region in roi_masks.items()
    }

    full_depth = roi_statistics["full"].median
    top_depth = roi_statistics["top"].median
    lower_depth = roi_statistics["lower"].median
    top_lower_depth_difference = relative_depth_difference(top_depth, lower_depth)

    selected_depth: float | None
    selected_roi_name: str
    quality: DepthQuality
    if (
        top_lower_depth_difference is not None
        and top_lower_depth_difference <= HIGH_QUALITY_DIFFERENCE
    ):
        selected_depth = lower_depth
        selected_roi_name = "lower"
        quality = "high"
    elif (
        top_lower_depth_difference is not None
        and top_lower_depth_difference <= MEDIUM_QUALITY_DIFFERENCE
    ):
        selected_depth = _finite_median((full_depth, top_depth, lower_depth))
        selected_roi_name = "full_top_lower_median"
        quality = "medium"
    else:
        selected_depth = full_depth
        selected_roi_name = "full"
        quality = "low"

    if selected_depth is None:
        selected_depth, selected_roi_name = _first_valid_roi(
            roi_statistics,
            order=("full", "top", "lower", "foot"),
        )
    if selected_depth is None:
        quality = "unavailable"
        selected_roi_name = "unavailable"

    return RepresentativeDepth(
        value=selected_depth,
        source=f"{base_source}:{selected_roi_name}",
        quality=quality,
        roi_statistics=roi_statistics,
        top_lower_relative_difference=top_lower_depth_difference,
    )


def load_representative_depth(
    detection: Detection,
    depth_map: NDArray[np.generic],
    *,
    config: RepresentativeDepthConfig | None = None,
) -> RepresentativeDepth:
    """Use a hanging object's inner ROI before a full-bbox fallback."""

    settings = config or RepresentativeDepthConfig()
    _validate_detection_class(detection, "hanging_object")
    _validate_depth_map(depth_map)
    bbox = _required_bbox(detection, depth_map.shape)
    inner_bbox = _centered_bbox(bbox, settings.load_inner_size_fraction)

    roi_statistics = {
        "inner": calculate_depth_statistics(
            _depth_roi(depth_map, inner_bbox),
            minimum_valid_pixels=settings.minimum_valid_pixels,
        ),
        "full": calculate_depth_statistics(
            _depth_roi(depth_map, bbox),
            minimum_valid_pixels=settings.minimum_valid_pixels,
        ),
    }
    selected_depth, selected_roi_name = _first_valid_roi(
        roi_statistics,
        order=("inner", "full"),
    )
    quality: DepthQuality = "high" if selected_roi_name == "inner" else "low"
    if selected_depth is None:
        selected_roi_name = "unavailable"
        quality = "unavailable"

    return RepresentativeDepth(
        value=selected_depth,
        source=selected_roi_name,
        quality=quality,
        roi_statistics=roi_statistics,
    )


def calculate_depth_statistics(
    values: NDArray[np.generic] | None,
    *,
    minimum_valid_pixels: int = 1,
) -> DepthStatistics:
    """Calculate finite relative-depth statistics for one ROI."""

    _validate_minimum_valid_pixels(minimum_valid_pixels)
    if values is None:
        return DepthStatistics(valid_count=0)

    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    valid_value_count = int(finite_values.size)
    if valid_value_count < minimum_valid_pixels:
        return DepthStatistics(valid_count=valid_value_count)

    return DepthStatistics(
        valid_count=valid_value_count,
        mean=float(np.mean(finite_values)),
        median=float(np.median(finite_values)),
        minimum=float(np.min(finite_values)),
        maximum=float(np.max(finite_values)),
        percentile_10=float(np.percentile(finite_values, 10)),
        percentile_90=float(np.percentile(finite_values, 90)),
        standard_deviation=float(np.std(finite_values)),
    )


def relative_depth_difference(
    depth_a: float | None,
    depth_b: float | None,
    *,
    epsilon: float = 1e-8,
) -> float | None:
    """Return a symmetric relative difference for two finite depths."""

    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and greater than zero")
    if not _is_finite(depth_a) or not _is_finite(depth_b):
        return None
    assert depth_a is not None and depth_b is not None
    return float(abs(depth_a - depth_b) / max(abs(depth_a), abs(depth_b), epsilon))


def _valid_detection_mask(
    detection: Detection,
    image_shape: tuple[int, ...],
) -> NDArray[np.bool_] | None:
    detection_mask = detection["mask"]
    expected_mask_shape = tuple(image_shape[:2])
    if (
        not isinstance(detection_mask, np.ndarray)
        or detection_mask.shape != expected_mask_shape
    ):
        return None
    boolean_mask = np.asarray(detection_mask, dtype=bool)
    return boolean_mask if np.any(boolean_mask) else None


def _required_bbox(
    detection: Detection,
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    bbox = clip_bbox(detection["bbox"], image_shape)
    if bbox is None:
        raise ValueError(f"{detection['class_name']} detection has an invalid bbox")
    return bbox


def _bbox_region(
    image_shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
) -> NDArray[np.bool_]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = _integer_bbox(bbox, image_shape)
    bbox_mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        bbox_mask[y1:y2, x1:x2] = True
    return bbox_mask


def _depth_roi(
    depth_map: NDArray[np.generic],
    bbox: tuple[float, float, float, float],
) -> NDArray[np.generic] | None:
    x1, y1, x2, y2 = _integer_bbox(bbox, depth_map.shape)
    depth_roi = depth_map[y1:y2, x1:x2]
    return depth_roi if depth_roi.size else None


def _integer_bbox(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    clipped = clip_bbox(bbox, image_shape)
    if clipped is None:
        return 0, 0, 0, 0
    height, width = image_shape[:2]
    x1 = max(0, min(width, int(np.floor(clipped[0]))))
    y1 = max(0, min(height, int(np.floor(clipped[1]))))
    x2 = max(0, min(width, int(np.ceil(clipped[2]))))
    y2 = max(0, min(height, int(np.ceil(clipped[3]))))
    return x1, y1, x2, y2


def _top_bbox(
    bbox: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2, y1 + (y2 - y1) * fraction


def _bottom_bbox(
    bbox: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, y2 - (y2 - y1) * fraction, x2, y2


def _centered_bbox(
    bbox: tuple[float, float, float, float],
    size_fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * size_fraction / 2.0
    half_height = (y2 - y1) * size_fraction / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _first_valid_roi(
    roi_statistics: dict[str, DepthStatistics],
    *,
    order: tuple[str, ...],
) -> tuple[float | None, str]:
    for roi_name in order:
        median_depth = roi_statistics[roi_name].median
        if median_depth is not None:
            return median_depth, roi_name
    return None, "unavailable"


def _finite_median(values: Iterable[float | None]) -> float | None:
    finite_values = [float(value) for value in values if _is_finite(value)]
    return float(np.median(finite_values)) if finite_values else None


def _is_finite(value: Any) -> bool:
    return value is not None and bool(np.isfinite(value))


def _validate_depth_map(depth_map: Any) -> None:
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("depth_map must be a numpy.ndarray")
    if depth_map.ndim != 2:
        raise ValueError("depth_map must have shape (height, width)")
    if depth_map.shape[0] <= 0 or depth_map.shape[1] <= 0:
        raise ValueError("depth_map height and width must be greater than zero")


def _validate_detection_class(detection: Detection, expected: str) -> None:
    if detection["class_name"] != expected:
        raise ValueError(
            f"expected a {expected} detection, got {detection['class_name']}"
        )


def _validate_minimum_valid_pixels(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("minimum_valid_pixels must be a positive integer")


__all__ = [
    "calculate_depth_statistics",
    "estimate_person_representative",
    "load_representative_depth",
    "person_representative_depth",
    "person_representative_point",
    "relative_depth_difference",
]
