from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import yaml

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR / "packages"))

from vision_engine.frame_pipeline import VisionFramePipeline
from vision_engine.model_manager import (
    ModelManager,
    build_model_manager,
)


class LoadableModel:
    def __init__(self):
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return self

    def metadata(self):
        return {"loaded": self.load_calls > 0}


class ModelManagerTests(unittest.TestCase):
    def test_factory_and_load_are_called_only_once(self):
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return LoadableModel()

        manager = ModelManager()
        manager.register("model", factory)

        first = manager.get("model")
        second = manager.get("model")

        self.assertIs(first, second)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(first.load_calls, 1)


@unittest.skipUnless(
    os.getenv("RUN_MODEL_INTEGRATION") == "1",
    "set RUN_MODEL_INTEGRATION=1 to load and run real models",
)
class RealModelIntegrationTests(unittest.TestCase):
    """Opt-in smoke test for the three real Phase-1 models."""

    def test_real_models_on_one_image(self):
        config_path = Path(
            os.getenv(
                "CRANE_MODEL_CONFIG",
                str(PROJECT_DIR / "configs" / "models.local.yaml"),
            )
        ).resolve()
        image_value = os.getenv("CRANE_TEST_IMAGE")
        if image_value is None:
            self.skipTest("set CRANE_TEST_IMAGE to an input image path")
        image_path = Path(image_value).resolve()

        self.assertTrue(config_path.is_file(), f"missing config: {config_path}")
        self.assertTrue(image_path.is_file(), f"missing image: {image_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertIsInstance(config, dict)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, f"OpenCV could not decode: {image_path}")

        manager = build_model_manager(config, config_dir=config_path.parent)
        pipeline = VisionFramePipeline(manager)
        result = pipeline.process(
            image,
            frame_id="integration-frame-000001",
            image_path=image_path,
        )
        metadata = manager.metadata()

        detection_summary = [
            {
                "class_name": detection["class_name"],
                "confidence": round(detection["confidence"], 4),
                "bbox": list(detection["bbox"]),
                "has_mask": detection["mask"] is not None,
                "mask_shape": (
                    list(detection["mask"].shape)
                    if detection["mask"] is not None
                    else None
                ),
            }
            for detection in result.detections
        ]
        print("\nReal model metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("\nReal detection summary:")
        print(json.dumps(detection_summary, indent=2, ensure_ascii=False))
        print("\nRelative-depth metadata:")
        print(
            json.dumps(
                result.relative_depth.metadata.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

        self.assertTrue(metadata["models"]["rfdetr"]["loaded"])
        self.assertTrue(metadata["models"]["yolo_person"]["loaded"])
        self.assertTrue(metadata["models"]["depth_anything_v3"]["loaded"])
        self.assertEqual(
            metadata["models"]["depth_anything_v3"]["inference_count"], 1
        )
        self.assertEqual(result.relative_depth.depth_map.shape, image.shape[:2])


if __name__ == "__main__":
    unittest.main()
