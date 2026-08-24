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
    GeometryFrameResult,
    ImagePoint,
    LoadAnchorCandidate,
    LoadAnchorCandidates,
    LoadGeometryResult,
    LoadSafetyZones,
    PersonGeometryResult,
    PersonRepresentative,
    PseudoBEVPoint,
    PseudoBEVRectangle,
    RepresentativeDepth,
)
from .depth_utils import calculate_depth_normalization_range
from .farthest_point_sampling import select_farthest_load_anchors
from .frame_pipeline import GeometryFramePipeline
from .load_anchors import (
    build_load_anchor_candidates,
    filter_candidates_by_seed_depth,
    generate_candidate_patches,
)
from .pseudo_bev import (
    project_image_point_to_pseudo_bev,
    project_load_anchors_to_pseudo_bev,
    project_person_to_pseudo_bev,
    relative_depth_to_forward,
)
from .representative_depth import (
    calculate_depth_statistics,
    estimate_person_representative,
    load_representative_depth,
    person_representative_depth,
    person_representative_point,
    relative_depth_difference,
)
from .zones import (
    build_load_footprint,
    build_load_zones,
    expand_footprint,
    rectangle_from_center_and_half_size,
)

__all__ = [
    "ConnectedRegionConfig",
    "DepthQuality",
    "DepthStatistics",
    "FarthestPointSamplingConfig",
    "GeometryConfig",
    "GeometryFramePipeline",
    "GeometryFrameResult",
    "ImagePoint",
    "LoadAnchorCandidate",
    "LoadAnchorCandidates",
    "LoadAnchorsConfig",
    "LoadGeometryResult",
    "LoadSafetyZones",
    "PersonGeometryResult",
    "PersonRepresentative",
    "PseudoBEVConfig",
    "PseudoBEVPoint",
    "PseudoBEVRectangle",
    "RepresentativeDepth",
    "RepresentativeDepthConfig",
    "ZoneBufferConfig",
    "ZonesConfig",
    "build_load_anchor_candidates",
    "build_load_footprint",
    "build_load_zones",
    "calculate_depth_normalization_range",
    "calculate_depth_statistics",
    "estimate_person_representative",
    "expand_footprint",
    "filter_candidates_by_seed_depth",
    "find_connected_candidate_region",
    "generate_candidate_patches",
    "load_geometry_config",
    "load_representative_depth",
    "person_representative_depth",
    "person_representative_point",
    "project_image_point_to_pseudo_bev",
    "project_load_anchors_to_pseudo_bev",
    "project_person_to_pseudo_bev",
    "rectangle_from_center_and_half_size",
    "relative_depth_difference",
    "relative_depth_to_forward",
    "select_farthest_load_anchors",
]
