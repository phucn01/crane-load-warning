"""Reusable computer-vision components for the crane safety backend."""

from .contracts import (
    DepthMapMetadata,
    Detection,
    RelativeDepthResult,
    VisionFrameResult,
)
from .frame_pipeline import (
    VisionFramePipeline,
    write_vision_artifacts,
)
from .model_manager import ModelManager, build_model_manager

__all__ = [
    "DepthMapMetadata",
    "Detection",
    "ModelManager",
    "RelativeDepthResult",
    "VisionFramePipeline",
    "VisionFrameResult",
    "build_model_manager",
    "write_vision_artifacts",
]
