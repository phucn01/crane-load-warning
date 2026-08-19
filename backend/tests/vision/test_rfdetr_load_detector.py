from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.detectors import RFDETRLoadDetector  # noqa: E402


class FakeRFDETR:
    def __init__(self):
        self.calls = 0
        self.threshold = None
        self.image = None

    def predict(self, image, threshold):
        self.calls += 1
        self.image = image
        self.threshold = threshold
        return SimpleNamespace(
            xyxy=np.array([[-2, 1, 8, 9], [2, 2, 5, 7]], dtype=np.float32),
            confidence=np.array([0.8, 0.7], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int64),
        )


class RFDETRLoadDetectorTests(unittest.TestCase):
    def test_maps_classes_and_clips_boxes(self):
        model = FakeRFDETR()
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        detections = RFDETRLoadDetector(model=model).predict(image)

        self.assertEqual([item["class_name"] for item in detections], [
            "hanging_object",
            "hanging_rope",
        ])
        self.assertEqual(detections[0]["bbox"], (0.0, 1.0, 8.0, 9.0))
        self.assertIsNone(detections[0]["mask"])
        self.assertAlmostEqual(model.threshold, 0.25)
        self.assertIsInstance(model.image, Image.Image)


if __name__ == "__main__":
    unittest.main()
