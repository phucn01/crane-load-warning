from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.contracts import clip_bbox, detection_to_dict  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_bbox_clipping_rejects_invalid_and_clips_to_image(self):
        self.assertEqual(
            clip_bbox((-5, 2, 120, 80), (50, 100, 3)),
            (0.0, 2.0, 100.0, 50.0),
        )
        self.assertIsNone(clip_bbox((5, 5, 5, 8), (50, 100, 3)))
        self.assertIsNone(clip_bbox((np.nan, 1, 5, 8), (50, 100, 3)))

    def test_detection_serialization_uses_mask_reference(self):
        detection = {
            "source_model": "YOLO26m",
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.9,
            "bbox": (1.0, 2.0, 8.0, 9.0),
            "x1": 1.0,
            "y1": 2.0,
            "x2": 8.0,
            "y2": 9.0,
            "mask": np.ones((10, 10), dtype=bool),
        }

        payload = detection_to_dict(
            detection, detection_id="person_01", mask_ref="masks/person_01.npy"
        )

        self.assertEqual(payload["bbox"]["format"], "xyxy_absolute")
        self.assertEqual(payload["mask_ref"], "masks/person_01.npy")
        self.assertTrue(payload["has_mask"])
        self.assertNotIn("mask", payload)


if __name__ == "__main__":
    unittest.main()
