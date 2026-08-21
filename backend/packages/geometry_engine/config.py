"""Typed configuration for the Phase-2 geometry pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite value greater than zero")


def _require_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative value")


def _require_fraction(
    name: str,
    value: float,
    *,
    allow_zero: bool = True,
    upper: float = 1.0,
) -> None:
    lower_is_valid = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not lower_is_valid or value > upper:
        lower = "0" if allow_zero else "0 (exclusive)"
        raise ValueError(f"{name} must be between {lower} and {upper}")


@dataclass(frozen=True, slots=True)
class RepresentativeDepthConfig:
    """Parameters for selecting representative person/load depths."""

    person_bottom_fraction: float = 0.10
    minimum_valid_pixels: int = 5
    load_inner_size_fraction: float = 0.20

    def __post_init__(self) -> None:
        _require_fraction(
            "representative_depth.person_bottom_fraction",
            self.person_bottom_fraction,
            allow_zero=False,
        )
        _require_positive_int(
            "representative_depth.minimum_valid_pixels",
            self.minimum_valid_pixels,
        )
        _require_fraction(
            "representative_depth.load_inner_size_fraction",
            self.load_inner_size_fraction,
            allow_zero=False,
        )


@dataclass(frozen=True, slots=True)
class LoadAnchorsConfig:
    """Parameters for generating and filtering load candidate patches."""

    patch_size: int = 11
    patch_stride: int = 12
    seed_depth_tolerance: float = 0.30

    def __post_init__(self) -> None:
        _require_positive_int("load_anchors.patch_size", self.patch_size)
        if self.patch_size % 2 == 0:
            raise ValueError("load_anchors.patch_size must be odd")
        _require_positive_int("load_anchors.patch_stride", self.patch_stride)
        _require_non_negative(
            "load_anchors.seed_depth_tolerance",
            self.seed_depth_tolerance,
        )

    @property
    def patch_radius(self) -> int:
        """Radius in pixels on either side of a patch center."""

        return self.patch_size // 2


@dataclass(frozen=True, slots=True)
class ConnectedRegionConfig:
    """Local-depth and neighborhood settings for region growing."""

    neighbor_radius: int = 1
    local_neighbor_depth_tolerance: float = 0.05

    def __post_init__(self) -> None:
        _require_positive_int(
            "connected_region.neighbor_radius",
            self.neighbor_radius,
        )
        _require_non_negative(
            "connected_region.local_neighbor_depth_tolerance",
            self.local_neighbor_depth_tolerance,
        )


@dataclass(frozen=True, slots=True)
class FarthestPointSamplingConfig:
    """Spatial coverage settings for final anchor selection."""

    maximum_anchors: int = 16
    minimum_distance: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_int(
            "farthest_point_sampling.maximum_anchors",
            self.maximum_anchors,
        )
        _require_non_negative(
            "farthest_point_sampling.minimum_distance",
            self.minimum_distance,
        )


@dataclass(frozen=True, slots=True)
class PseudoBEVConfig:
    """Scaling and numerical safety settings for relative Pseudo-BEV."""

    lateral_scale: float = 1.0
    longitudinal_scale: float = 1.0
    minimum_depth: float = 1e-6

    def __post_init__(self) -> None:
        _require_positive("pseudo_bev.lateral_scale", self.lateral_scale)
        _require_positive(
            "pseudo_bev.longitudinal_scale",
            self.longitudinal_scale,
        )
        _require_positive("pseudo_bev.minimum_depth", self.minimum_depth)


@dataclass(frozen=True, slots=True)
class ZoneBufferConfig:
    """Per-axis expansion ratios relative to a load footprint half-size."""

    lateral_ratio: float
    longitudinal_ratio: float

    def __post_init__(self) -> None:
        _require_non_negative("zone lateral ratio", self.lateral_ratio)
        _require_non_negative("zone longitudinal ratio", self.longitudinal_ratio)


@dataclass(frozen=True, slots=True)
class ZonesConfig:
    """Sequential footprint-to-danger and danger-to-warning expansion ratios."""

    danger: ZoneBufferConfig = ZoneBufferConfig(
        lateral_ratio=0.15,
        longitudinal_ratio=0.20,
    )
    warning: ZoneBufferConfig = ZoneBufferConfig(
        lateral_ratio=0.30,
        longitudinal_ratio=0.40,
    )

@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Complete validated configuration for the geometry pipeline."""

    representative_depth: RepresentativeDepthConfig = RepresentativeDepthConfig()
    load_anchors: LoadAnchorsConfig = LoadAnchorsConfig()
    connected_region: ConnectedRegionConfig = ConnectedRegionConfig()
    farthest_point_sampling: FarthestPointSamplingConfig = FarthestPointSamplingConfig()
    pseudo_bev: PseudoBEVConfig = PseudoBEVConfig()
    zones: ZonesConfig = ZonesConfig()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GeometryConfig:
        """Build a config from a parsed YAML mapping and reject unknown keys."""

        _reject_unknown_keys(
            "geometry config",
            payload,
            {
                "representative_depth",
                "load_anchors",
                "connected_region",
                "farthest_point_sampling",
                "pseudo_bev",
                "zones",
            },
        )

        return cls(
            representative_depth=RepresentativeDepthConfig(
                **_section(payload, "representative_depth")
            ),
            load_anchors=LoadAnchorsConfig(**_section(payload, "load_anchors")),
            connected_region=ConnectedRegionConfig(
                **_section(payload, "connected_region")
            ),
            farthest_point_sampling=FarthestPointSamplingConfig(
                **_section(payload, "farthest_point_sampling")
            ),
            pseudo_bev=PseudoBEVConfig(**_section(payload, "pseudo_bev")),
            zones=_parse_zones(_section(payload, "zones")),
        )


def load_geometry_config(path: str | Path) -> GeometryConfig:
    """Read and validate a geometry YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)

    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise TypeError("geometry config root must be a YAML mapping")
    return GeometryConfig.from_mapping(payload)


def _parse_zones(payload: Mapping[str, Any]) -> ZonesConfig:
    _reject_unknown_keys("zones", payload, {"danger", "warning"})
    default_zones = ZonesConfig()
    danger_mapping = _section(payload, "danger")
    warning_mapping = _section(payload, "warning")
    _reject_unknown_keys(
        "zones.danger",
        danger_mapping,
        {"lateral_ratio", "longitudinal_ratio"},
    )
    _reject_unknown_keys(
        "zones.warning",
        warning_mapping,
        {"lateral_ratio", "longitudinal_ratio"},
    )
    return ZonesConfig(
        danger=ZoneBufferConfig(
            lateral_ratio=float(
                danger_mapping.get(
                    "lateral_ratio",
                    default_zones.danger.lateral_ratio,
                )
            ),
            longitudinal_ratio=float(
                danger_mapping.get(
                    "longitudinal_ratio",
                    default_zones.danger.longitudinal_ratio,
                )
            ),
        ),
        warning=ZoneBufferConfig(
            lateral_ratio=float(
                warning_mapping.get(
                    "lateral_ratio",
                    default_zones.warning.lateral_ratio,
                )
            ),
            longitudinal_ratio=float(
                warning_mapping.get(
                    "longitudinal_ratio",
                    default_zones.warning.longitudinal_ratio,
                )
            ),
        ),
    )


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _reject_unknown_keys(
    section_name: str,
    payload: Mapping[str, Any],
    allowed_keys: set[str],
) -> None:
    unknown_keys = sorted(str(key) for key in payload if key not in allowed_keys)
    if unknown_keys:
        unknown_key_names = ", ".join(unknown_keys)
        raise ValueError(f"unknown keys in {section_name}: {unknown_key_names}")


__all__ = [
    "ConnectedRegionConfig",
    "FarthestPointSamplingConfig",
    "GeometryConfig",
    "LoadAnchorsConfig",
    "PseudoBEVConfig",
    "RepresentativeDepthConfig",
    "ZoneBufferConfig",
    "ZonesConfig",
    "load_geometry_config",
]
