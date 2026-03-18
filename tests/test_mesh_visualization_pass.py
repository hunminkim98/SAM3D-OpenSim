import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda *args, **kwargs: None,
    )

from src.sam3d_inference import SAM3DInference


class _FakeEstimator:
    def __init__(self, outputs):
        self.outputs = outputs
        self.fov_estimator = "original"

    def process_one_image(self, *args, **kwargs):
        return self.outputs


class MeshVisualizationPassTests(unittest.TestCase):
    def test_process_frame_can_bypass_focal_cache(self):
        inference = SAM3DInference.__new__(SAM3DInference)
        inference.per_frame_logging = True
        inference.bbox_threshold = 0.8
        inference.nms_threshold = 0.3
        inference.use_mask = False
        inference.inference_type = "full"
        inference._cached_focal_length = 999.0
        inference.estimator = _FakeEstimator(
            [{"focal_length": 123.0, "pred_cam_t": [0.0, 0.0, 1.0]}]
        )

        outputs = inference.process_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            frame_idx=0,
            bboxes=None,
            use_cached_focal_length=False,
            force_fov_estimator="forced",
        )

        self.assertEqual(outputs[0]["focal_length"], 123.0)
        self.assertEqual(inference._cached_focal_length, 999.0)
        self.assertEqual(inference.estimator.fov_estimator, "original")

    def test_process_video_for_mesh_sidecars_uses_full_refresh_outputs(self):
        inference = SAM3DInference.__new__(SAM3DInference)
        inference.single_person = True
        inference.selected_bbox_first_frame = [10.0, 10.0, 20.0, 20.0]
        inference.selected_person_index_first_frame = 0
        inference.mesh_visualization_fov_estimator = "mesh_fov"

        process_calls = []
        per_frame_outputs = [
            [
                {"bbox": [100.0, 100.0, 140.0, 140.0], "pred_cam_t": [0, 0, 4]},
                {"bbox": [10.0, 10.0, 20.0, 20.0], "pred_cam_t": [0, 0, 3]},
            ],
            [
                {"bbox": [11.0, 10.0, 21.0, 20.0], "pred_cam_t": [0, 0, 3]},
                {"bbox": [200.0, 200.0, 240.0, 240.0], "pred_cam_t": [0, 0, 5]},
            ],
        ]

        def fake_process_frame(image, frame_idx=0, bboxes=None, **kwargs):
            process_calls.append({"frame_idx": frame_idx, "bboxes": bboxes, **kwargs})
            return per_frame_outputs[frame_idx]

        inference.process_frame = fake_process_frame

        with patch("cv2.imread", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            with patch("cv2.cvtColor", side_effect=lambda image, _code: image):
                outputs = inference.process_video_for_mesh_sidecars(
                    ["frame_0.jpg", "frame_1.jpg"],
                    progress=False,
                )

        self.assertEqual(outputs[0]["selected_output_index"], 1)
        self.assertEqual(outputs[1]["selected_output_index"], 0)
        self.assertEqual(process_calls[0]["bboxes"], None)
        self.assertEqual(process_calls[1]["bboxes"], None)
        self.assertFalse(process_calls[0]["use_cached_focal_length"])
        self.assertFalse(process_calls[1]["use_cached_focal_length"])
        self.assertEqual(process_calls[0]["force_fov_estimator"], "mesh_fov")
        self.assertEqual(process_calls[1]["force_fov_estimator"], "mesh_fov")


if __name__ == "__main__":
    unittest.main()
