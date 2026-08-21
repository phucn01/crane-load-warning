"""Typed runtime contracts for relative geometry processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DepthQuality = Literal["high", "medium", "low", "unavailable"]


@dataclass(frozen=True, slots=True)
class ImagePoint:
    """A point in absolute image pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PseudoBEVPoint:
    """A non-metric point in relative lateral/longitudinal coordinates."""

    lateral: float
    longitudinal: float


@dataclass(frozen=True, slots=True)
class PseudoBEVRectangle:
    """Axis-aligned bounds in relative Pseudo-BEV coordinates."""

    minimum_lateral: float
    maximum_lateral: float
    minimum_longitudinal: float
    maximum_longitudinal: float

    @property
    def center_lateral(self) -> float:
        return (self.minimum_lateral + self.maximum_lateral) / 2.0

    @property
    def center_longitudinal(self) -> float:
        return (self.minimum_longitudinal + self.maximum_longitudinal) / 2.0

    @property
    def half_lateral(self) -> float:
        return (self.maximum_lateral - self.minimum_lateral) / 2.0

    @property
    def half_longitudinal(self) -> float:
        return (self.maximum_longitudinal - self.minimum_longitudinal) / 2.0

    @property
    def corners(self) -> tuple[PseudoBEVPoint, ...]:
        """Return corners in clockwise order from the lower-left corner."""

        return (
            PseudoBEVPoint(self.minimum_lateral, self.minimum_longitudinal),
            PseudoBEVPoint(self.minimum_lateral, self.maximum_longitudinal),
            PseudoBEVPoint(self.maximum_lateral, self.maximum_longitudinal),
            PseudoBEVPoint(self.maximum_lateral, self.minimum_longitudinal),
        )


@dataclass(frozen=True, slots=True)
class LoadSafetyZones:
    """A load footprint and its nested danger and warning rectangles."""

    footprint: PseudoBEVRectangle
    danger: PseudoBEVRectangle
    warning: PseudoBEVRectangle


@dataclass(frozen=True, slots=True)
class DepthStatistics:
    """Finite relative-depth statistics for one ROI."""

    valid_count: int
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    percentile_10: float | None = None
    percentile_90: float | None = None
    standard_deviation: float | None = None

    @property
    def is_valid(self) -> bool:
        return self.median is not None


@dataclass(frozen=True, slots=True)
class RepresentativeDepth:
    """Selected relative depth plus diagnostic ROI information."""

    value: float | None
    source: str
    quality: DepthQuality
    roi_statistics: dict[str, DepthStatistics] = field(default_factory=dict)
    top_lower_relative_difference: float | None = None


@dataclass(frozen=True, slots=True)
class PersonRepresentative:
    """Representative image position and relative depth for one person."""

    point: ImagePoint
    point_source: str
    depth: RepresentativeDepth


@dataclass(frozen=True, slots=True)
class LoadAnchorCandidate:
    """One valid-depth patch sampled inside a hanging-object bbox."""

    candidate_id: str
    grid_x: int
    grid_y: int
    point: ImagePoint
    patch_bbox: tuple[int, int, int, int]
    depth: float
    valid_depth_count: int
    valid_depth_fraction: float
    seed_depth_difference: float | None = None
    is_seed_consistent: bool = False


@dataclass(frozen=True, slots=True)
class LoadAnchorCandidates:
    """Candidate generation and seed-depth filtering result for one load."""

    seed_depth: float | None
    seed_source: str
    generated_patch_count: int
    rejected_invalid_depth_count: int
    candidates: tuple[LoadAnchorCandidate, ...]

    @property
    def consistent_candidates(self) -> tuple[LoadAnchorCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate.is_seed_consistent
        )


__all__ = [
    "DepthQuality",
    "DepthStatistics",
    "ImagePoint",
    "LoadAnchorCandidate",
    "LoadAnchorCandidates",
    "LoadSafetyZones",
    "PersonRepresentative",
    "PseudoBEVPoint",
    "PseudoBEVRectangle",
    "RepresentativeDepth",
]
