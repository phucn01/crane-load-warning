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
    "PersonRepresentative",
    "RepresentativeDepth",
]
