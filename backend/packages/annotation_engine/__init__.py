"""Offline annotation and safety evidence generation."""

from .contracts import EvidenceArtifacts, EvidenceTraceability
from .evidence_composer import (
    OfflineEvidenceComposer,
    build_assessment_payload,
    compose_evidence_image,
)
from .image_overlay import (
    render_image_overlay,
    render_safe_no_load_overlay,
    render_skipped_overlay,
)
from .pseudo_bev_overlay import (
    draw_pseudo_bev_chart,
    render_pseudo_bev_chart,
    render_pseudo_bev_overlay,
)

__all__ = [
    "EvidenceArtifacts",
    "EvidenceTraceability",
    "OfflineEvidenceComposer",
    "build_assessment_payload",
    "compose_evidence_image",
    "draw_pseudo_bev_chart",
    "render_image_overlay",
    "render_pseudo_bev_chart",
    "render_pseudo_bev_overlay",
    "render_safe_no_load_overlay",
    "render_skipped_overlay",
]
