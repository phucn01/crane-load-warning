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
from .connected_region import find_connected_candidate_region
from .contracts import (
    DepthQuality,
    DepthStatistics,
    ImagePoint,
    LoadAnchorCandidate,
    LoadAnchorCandidates,
    PersonRepresentative,
    RepresentativeDepth,
)
from .farthest_point_sampling import select_farthest_load_anchors
from .load_anchors import (
    build_load_anchor_candidates,
    filter_candidates_by_seed_depth,
    generate_candidate_patches,
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
    "LoadAnchorCandidate",
    "LoadAnchorCandidates",
    "LoadAnchorsConfig",
    "PersonRepresentative",
    "PseudoBEVConfig",
    "RepresentativeDepth",
    "RepresentativeDepthConfig",
    "ZoneBufferConfig",
    "ZonesConfig",
    "build_load_anchor_candidates",
    "calculate_depth_statistics",
    "estimate_person_representative",
    "filter_candidates_by_seed_depth",
    "find_connected_candidate_region",
    "generate_candidate_patches",
    "load_geometry_config",
    "load_representative_depth",
    "person_representative_depth",
    "person_representative_point",
    "relative_depth_difference",
    "select_farthest_load_anchors",
]
