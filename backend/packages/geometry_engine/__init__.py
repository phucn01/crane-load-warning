"""Relative geometry processing for crane load safety."""

from .config import (
    ConnectedRegionConfig,
    FarthestPointSamplingConfig,
    GeometryConfig,
    LoadAnchorsConfig,
    PseudoBEVConfig,
    RepresentativeDepthConfig,
    ZoneBufferConfig,
    ZonesConfig,
    load_geometry_config,
)
from .contracts import (
    DepthQuality,
    DepthStatistics,
    ImagePoint,
    PersonRepresentative,
    RepresentativeDepth,
)
from .representative_depth import (
    calculate_depth_statistics,
    estimate_person_representative,
    load_representative_depth,
    person_representative_depth,
    person_representative_point,
    relative_depth_difference,
)

__all__ = [
    "ConnectedRegionConfig",
    "DepthQuality",
    "DepthStatistics",
    "FarthestPointSamplingConfig",
    "GeometryConfig",
    "ImagePoint",
    "LoadAnchorsConfig",
    "PersonRepresentative",
    "PseudoBEVConfig",
    "RepresentativeDepth",
    "RepresentativeDepthConfig",
    "ZoneBufferConfig",
    "ZonesConfig",
    "calculate_depth_statistics",
    "estimate_person_representative",
    "load_geometry_config",
    "load_representative_depth",
    "person_representative_depth",
    "person_representative_point",
    "relative_depth_difference",
]
