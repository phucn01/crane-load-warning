"""Project image points and relative depth into a non-metric Pseudo-BEV."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .config import PseudoBEVConfig
from .contracts import (
    ImagePoint,
    LoadAnchorCandidate,
    PersonRepresentative,
    PseudoBEVPoint,
)


def project_image_point_to_pseudo_bev(
    point: ImagePoint,
    depth: float | None,
    *,
    image_width: int,
    depth_low: float,
    depth_high: float,
    config: PseudoBEVConfig | None = None,
) -> PseudoBEVPoint | None:
    """Project one image point into centered lateral/depth coordinates.

    The image center is lateral zero and horizontal displacement is normalized
    by image width without forward-depth multiplication. Scene-normalized
    relative depth supplies the longitudinal coordinate. The result is useful
    for comparisons within the same camera view, but is not a metric world
    coordinate.
    """

    settings = config or PseudoBEVConfig()
    _validate_image_width(image_width)
    if not _is_finite_point(point) or depth is None or not math.isfinite(depth):
        return None

    forward = relative_depth_to_forward(float(depth), depth_low, depth_high)
    if not math.isfinite(forward):
        return None

    image_center_x = image_width / 2.0
    normalized_horizontal_offset = (point.x - image_center_x) / image_width
    lateral = normalized_horizontal_offset * settings.lateral_scale
    longitudinal = forward * settings.longitudinal_scale

    return PseudoBEVPoint(
        lateral=lateral,
        longitudinal=_clamp_unit_interval(longitudinal),
    )


def project_load_anchors_to_pseudo_bev(
    anchors: Sequence[LoadAnchorCandidate],
    *,
    image_width: int,
    depth_low: float,
    depth_high: float,
    config: PseudoBEVConfig | None = None,
) -> tuple[PseudoBEVPoint, ...]:
    """Project finite final load anchors while preserving their input order."""

    projected_points = (
        project_image_point_to_pseudo_bev(
            anchor.point,
            anchor.depth,
            image_width=image_width,
            depth_low=depth_low,
            depth_high=depth_high,
            config=config,
        )
        for anchor in anchors
    )
    return tuple(point for point in projected_points if point is not None)


def project_person_to_pseudo_bev(
    person: PersonRepresentative,
    *,
    image_width: int,
    depth_low: float,
    depth_high: float,
    config: PseudoBEVConfig | None = None,
) -> PseudoBEVPoint | None:
    """Project a person's representative point and selected relative depth."""

    return project_image_point_to_pseudo_bev(
        person.point,
        person.depth.value,
        image_width=image_width,
        depth_low=depth_low,
        depth_high=depth_high,
        config=config,
    )


def relative_depth_to_forward(
    depth_value: float,
    depth_low: float,
    depth_high: float,
) -> float:
    """Normalize scene-relative depth to the closed forward interval [0, 1]."""

    if not np.isfinite(depth_value):
        return float("nan")
    denominator = max(depth_high - depth_low, 1e-8)
    return float(np.clip((depth_value - depth_low) / denominator, 0.0, 1.0))


def _validate_image_width(image_width: int) -> None:
    if isinstance(image_width, bool) or not isinstance(image_width, int):
        raise TypeError("image_width must be an integer")
    if image_width <= 0:
        raise ValueError("image_width must be greater than zero")


def _is_finite_point(point: ImagePoint) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)


def _clamp_unit_interval(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


__all__ = [
    "project_image_point_to_pseudo_bev",
    "project_load_anchors_to_pseudo_bev",
    "project_person_to_pseudo_bev",
    "relative_depth_to_forward",
]
