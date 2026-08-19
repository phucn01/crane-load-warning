"""Object detection and segmentation adapters."""

from .yolo_person_segmenter import (
    PersonSegmentation,
    YOLOPersonSegmenter,
    YOLOPersonSegmenterConfig,
    run_person_detection,
)
from .rfdetr_load_detector import RFDETRLoadDetector, RFDETRLoadDetectorConfig

__all__ = [
    "PersonSegmentation",
    "YOLOPersonSegmenter",
    "YOLOPersonSegmenterConfig",
    "run_person_detection",
    "RFDETRLoadDetector",
    "RFDETRLoadDetectorConfig",
]
