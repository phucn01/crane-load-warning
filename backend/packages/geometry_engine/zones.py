"""Build load footprints and buffered safety zones in Pseudo-BEV space."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import PseudoBEVConfig, ZoneBufferConfig, ZonesConfig
from .contracts import LoadSafetyZones, PseudoBEVPoint, PseudoBEVRectangle


def build_load_footprint(
    final_anchors: Sequence[PseudoBEVPoint],
    *,
    load_bbox: tuple[float, float, float, float],
    image_width: int,
    pseudo_bev_config: PseudoBEVConfig | None = None,
) -> PseudoBEVRectangle | None:
    """Use bbox lateral bounds and final-anchor longitudinal min/max bounds."""

    valid_anchors = tuple(point for point in final_anchors if _is_finite(point))
    if len(valid_anchors) < 2:
        return None

    # Current strategy: use the detected object's full bbox for lateral bounds.
    minimum_lateral, maximum_lateral = _bbox_lateral_bounds(
        load_bbox,
        image_width=image_width,
        config=pseudo_bev_config,
    )

    # Alternative anchor-based strategy:
    # lateral_values = tuple(point.lateral for point in valid_anchors)
    # minimum_lateral = min(lateral_values)
    # maximum_lateral = max(lateral_values)
    longitudinal_values = tuple(point.longitudinal for point in valid_anchors)
    minimum_longitudinal = min(longitudinal_values)
    maximum_longitudinal = max(longitudinal_values)

    return rectangle_from_center_and_half_size(
        center_lateral=(minimum_lateral + maximum_lateral) / 2.0,
        center_longitudinal=(minimum_longitudinal + maximum_longitudinal) / 2.0,
        half_lateral=(maximum_lateral - minimum_lateral) / 2.0,
        half_longitudinal=(maximum_longitudinal - minimum_longitudinal) / 2.0,
    )


def rectangle_from_center_and_half_size(
    *,
    center_lateral: float,
    center_longitudinal: float,
    half_lateral: float,
    half_longitudinal: float,
) -> PseudoBEVRectangle:
    """Create axis-aligned bounds from a center point and non-negative halves."""

    values = (center_lateral, center_longitudinal, half_lateral, half_longitudinal)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rectangle center and half-sizes must be finite")
    if half_lateral < 0.0 or half_longitudinal < 0.0:
        raise ValueError("rectangle half-sizes must be non-negative")

    return PseudoBEVRectangle(
        minimum_lateral=center_lateral - half_lateral,
        maximum_lateral=center_lateral + half_lateral,
        minimum_longitudinal=center_longitudinal - half_longitudinal,
        maximum_longitudinal=center_longitudinal + half_longitudinal,
    )


def expand_footprint(
    footprint: PseudoBEVRectangle,
    buffer: ZoneBufferConfig,
) -> PseudoBEVRectangle:
    """Expand each axis by a ratio of that footprint axis's half-size."""

    return rectangle_from_center_and_half_size(
        center_lateral=footprint.center_lateral,
        center_longitudinal=footprint.center_longitudinal,
        half_lateral=footprint.half_lateral * (1.0 + buffer.lateral_ratio),
        half_longitudinal=(
            footprint.half_longitudinal * (1.0 + buffer.longitudinal_ratio)
        ),
    )


def build_load_zones(
    final_anchors: Sequence[PseudoBEVPoint],
    *,
    load_bbox: tuple[float, float, float, float],
    image_width: int,
    pseudo_bev_config: PseudoBEVConfig | None = None,
    config: ZonesConfig | None = None,
) -> LoadSafetyZones | None:
    """Expand footprint to danger, then expand danger to warning for one load."""

    footprint = build_load_footprint(
        final_anchors,
        load_bbox=load_bbox,
        image_width=image_width,
        pseudo_bev_config=pseudo_bev_config,
    )
    if footprint is None:
        return None

    settings = config or ZonesConfig()
    danger = expand_footprint(footprint, settings.danger)
    return LoadSafetyZones(
        footprint=footprint,
        danger=danger,
        warning=expand_footprint(danger, settings.warning),
    )


def _is_finite(point: PseudoBEVPoint) -> bool:
    return math.isfinite(point.lateral) and math.isfinite(point.longitudinal)


def _bbox_lateral_bounds(
    load_bbox: tuple[float, float, float, float],
    *,
    image_width: int,
    config: PseudoBEVConfig | None,
) -> tuple[float, float]:
    if isinstance(image_width, bool) or not isinstance(image_width, int):
        raise TypeError("image_width must be an integer")
    if image_width <= 0:
        raise ValueError("image_width must be greater than zero")

    x1, _, x2, _ = load_bbox
    if not all(math.isfinite(value) for value in load_bbox):
        raise ValueError("load_bbox coordinates must be finite")
    if x2 < x1:
        raise ValueError("load_bbox x2 must be greater than or equal to x1")

    settings = config or PseudoBEVConfig()
    image_center_x = image_width / 2.0
    return (
        (x1 - image_center_x) / image_width * settings.lateral_scale,
        (x2 - image_center_x) / image_width * settings.lateral_scale,
    )


__all__ = [
    "build_load_footprint",
    "build_load_zones",
    "expand_footprint",
    "rectangle_from_center_and_half_size",
]
