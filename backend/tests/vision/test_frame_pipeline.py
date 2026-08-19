from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.contracts import (  # noqa: E402
    DepthMapMetadata,
    RelativeDepthResult,
)
from vision_engine.frame_pipeline import (  # noqa: E402
    OfflineFramePipeline,
    render_detection_preview,
    write_phase1_artifacts,
)


def make_detection(class_name, source_model, mask=None):
    return {
        "source_model": source_model,
        "class_id": 0,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox": (1.0, 1.0, 5.0, 5.0),
        "x1": 1.0,
        "y1": 1.0,
        "x2": 5.0,
        "y2": 5.0,
        "mask": mask,
    }


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return self.detections


class FakeDepth:
    def __init__(self):
        self.calls = 0

    def predict(self, image, image_path=None):
        self.calls += 1
        depth = np.ones(image.shape[:2], dtype=np.float32)
        return RelativeDepthResult(
            depth_map=depth,
            metadata=DepthMapMetadata(
                height=image.shape[0],
                width=image.shape[1],
                dtype="float32",
                finite_min=1.0,
                finite_max=1.0,
                finite_fraction=1.0,
            ),
        )


class FakeManager:
    def __init__(self, models):
        self.models = models

    def get(self, name):
        return self.models[name]


class FramePipelineTests(unittest.TestCase):
    def test_person_bbox_is_only_drawn_when_mask_is_missing(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        person = make_detection("person", "YOLO26m", mask)
        person["bbox"] = (20.0, 30.0, 80.0, 90.0)

        segmented_preview = render_detection_preview(image, [person])
        self.assertFalse(np.any(segmented_preview[30, 20]))
        self.assertTrue(np.any(segmented_preview[40, 40]))

        person_without_mask = dict(person)
        person_without_mask["mask"] = None
        fallback_preview = render_detection_preview(image, [person_without_mask])
        self.assertTrue(np.any(fallback_preview[30, 20]))

    def test_runs_every_component_once_and_writes_required_artifacts(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 2:5] = True
        load = FakeDetector([make_detection("hanging_object", "RF-DETR Medium")])
        person = FakeDetector([make_detection("person", "YOLO26m", mask)])
        depth = FakeDepth()
        pipeline = OfflineFramePipeline(
            FakeManager(
                {
                    "rfdetr": load,
                    "yolo_person": person,
                    "depth_anything_v3": depth,
                }
            )
        )

        result = pipeline.process(image)

        self.assertEqual((load.calls, person.calls, depth.calls), (1, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            write_phase1_artifacts(
                result,
                image_bgr=image,
                image_path="sample.png",
                output_dir=run_dir,
                model_metadata={"models": {}},
            )
            for name in (
                "detections.json",
                "relative_depth.npy",
                "model_metadata.json",
                "detection_preview.png",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            payload = json.loads(
                (run_dir / "detections.json").read_text(encoding="utf-8")
            )
            person_json = next(
                item for item in payload["detections"] if item["class_name"] == "person"
            )
            self.assertEqual(person_json["mask_ref"], "masks/person_01.npy")
            saved_mask = np.load(run_dir / person_json["mask_ref"], allow_pickle=False)
            np.testing.assert_array_equal(saved_mask, mask)


if __name__ == "__main__":
    unittest.main()
