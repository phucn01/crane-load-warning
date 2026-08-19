from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.depth import DepthAnythingV3  # noqa: E402


class FakeDepthModel:
    def __init__(self):
        self.calls = 0
        self.sources = None

    def inference(self, sources, process_res):
        self.calls += 1
        self.sources = sources
        return SimpleNamespace(
            depth=np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        )


class DepthAnythingV3Tests(unittest.TestCase):
    def test_produces_image_sized_float32_relative_depth_once(self):
        model = FakeDepthModel()
        estimator = DepthAnythingV3(model=model)
        image = np.zeros((4, 6, 3), dtype=np.uint8)

        result = estimator.predict(image)

        self.assertEqual(model.calls, 1)
        self.assertEqual(result.depth_map.shape, (4, 6))
        self.assertEqual(result.depth_map.dtype, np.float32)
        self.assertEqual(result.metadata.convention, "relative_depth_not_metric")
        self.assertEqual(estimator.metadata()["inference_count"], 1)

    def test_rejects_depth_without_finite_values(self):
        model = FakeDepthModel()
        model.inference = lambda sources, process_res: SimpleNamespace(
            depth=np.full((1, 2, 2), np.nan, dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "no finite"):
            DepthAnythingV3(model=model).predict(
                np.zeros((2, 2, 3), dtype=np.uint8)
            )


if __name__ == "__main__":
    unittest.main()
