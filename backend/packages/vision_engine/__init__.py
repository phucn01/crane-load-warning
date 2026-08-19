"""Reusable computer-vision components for the crane safety backend."""

from .contracts import (
    DepthMapMetadata,
    Detection,
    FramePipelineResult,
    RelativeDepthResult,
)
from .frame_pipeline import OfflineFramePipeline, write_phase1_artifacts
from .model_manager import ModelManager, build_model_manager

__all__ = [
    "DepthMapMetadata",
    "Detection",
    "FramePipelineResult",
    "ModelManager",
    "OfflineFramePipeline",
    "RelativeDepthResult",
    "build_model_manager",
    "write_phase1_artifacts",
]
