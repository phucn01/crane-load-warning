from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.detectors.yolo_person_segmenter import (  # noqa: E402
    YOLOPersonSegmenter,
    YOLOPersonSegmenterConfig,
)


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return self.results


class YOLOPersonSegmenterTests(unittest.TestCase):
    def test_keeps_bbox_confidence_and_matching_mask(self):
        boxes = SimpleNamespace(
            xyxy=np.array([[-5, 2, 8, 9], [1, 1, 4, 4]], dtype=np.float32),
            conf=np.array([0.91, 0.80], dtype=np.float32),
            cls=np.array([0, 2], dtype=np.float32),
        )
        masks = SimpleNamespace(
            xy=[
                np.array([[1, 2], [7, 2], [7, 8], [1, 8]], dtype=np.float32),
                np.array([[1, 1], [3, 1], [3, 3]], dtype=np.float32),
            ]
        )
        model = FakeModel([SimpleNamespace(boxes=boxes, masks=masks)])
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        detections = YOLOPersonSegmenter(model=model).predict(image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], (0.0, 2.0, 8.0, 9.0))
        self.assertAlmostEqual(detections[0]["confidence"], 0.91, places=5)
        self.assertEqual(detections[0]["mask"].shape, (10, 10))
        self.assertEqual(detections[0]["mask"].dtype, np.bool_)
        self.assertTrue(detections[0]["mask"][5, 5])
        self.assertEqual(model.kwargs["classes"], [0])
        self.assertTrue(model.kwargs["retina_masks"])

    def test_returns_none_mask_when_model_has_no_segmentation(self):
        boxes = SimpleNamespace(
            xyxy=np.array([[1, 2, 8, 9]], dtype=np.float32),
            conf=np.array([0.75], dtype=np.float32),
            cls=np.array([0], dtype=np.float32),
        )
        model = FakeModel([SimpleNamespace(boxes=boxes, masks=None)])

        detection = YOLOPersonSegmenter(model=model).predict(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )[0]

        self.assertIsNone(detection["mask"])
        self.assertEqual(detection["class_name"], "person")

    def test_configuration_is_forwarded_to_ultralytics(self):
        model = FakeModel([])
        config = YOLOPersonSegmenterConfig(
            confidence=0.42,
            image_size=960,
            device="cpu",
            retina_masks=False,
        )

        YOLOPersonSegmenter(model=model, config=config).predict(
            np.zeros((2, 3, 3), dtype=np.uint8)
        )

        self.assertEqual(model.kwargs["conf"], 0.42)
        self.assertEqual(model.kwargs["imgsz"], 960)
        self.assertEqual(model.kwargs["device"], "cpu")
        self.assertFalse(model.kwargs["retina_masks"])

    def test_rejects_invalid_image(self):
        segmenter = YOLOPersonSegmenter(model=FakeModel([]))
        with self.assertRaises(TypeError):
            segmenter.predict("image")

    def test_missing_official_weight_is_delegated_to_ultralytics_download(self):
        factory_sources = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models" / "yolo26m-seg.pt"

            def factory(source):
                factory_sources.append(source)
                return FakeModel([])

            segmenter = YOLOPersonSegmenter(target, model_factory=factory)
            segmenter.load()

            self.assertEqual(factory_sources, [str(target)])
            self.assertTrue(target.parent.is_dir())
            self.assertTrue(segmenter.metadata()["auto_download"])

    def test_missing_weight_fails_when_auto_download_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "yolo26m-seg.pt"
            config = YOLOPersonSegmenterConfig(auto_download=False)
            with self.assertRaises(FileNotFoundError):
                YOLOPersonSegmenter(
                    target,
                    config=config,
                    model_factory=lambda source: FakeModel([]),
                ).load()


if __name__ == "__main__":
    unittest.main()
