import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.export_stage import run_export_stage


class ExportStageIkBackendTests(unittest.TestCase):
    @patch("src.export_stage.load_video_outputs")
    @patch("src.export_stage.load_inference_meta")
    @patch("src.trc_exporter.TRCExporter.export")
    @patch("src.keypoint_converter.KeypointConverter.convert")
    @patch("src.moge_scene_ground.extract_scene_ground_arrays_from_json")
    @patch("src.post_ik_foot_snap.build_post_ik_contact_meta")
    @patch("src.coordinate_transform.CoordinateTransformer.get_last_contact_data")
    @patch("src.coordinate_transform.CoordinateTransformer.get_last_ground_alignment_info")
    @patch("src.coordinate_transform.CoordinateTransformer.transform")
    @patch("src.post_processing.PostProcessor.process")
    @patch("src.pose2sim_augmentation_runner.run_pose2sim_augmented_ik")
    @patch("src.opensim_ik.run_external_opensim_ik")
    def test_run_export_stage_routes_to_pose2sim_backend(
        self,
        direct_ik_mock,
        pose2sim_mock,
        process_mock,
        transform_mock,
        ground_info_mock,
        contact_data_mock,
        post_ik_meta_mock,
        scene_ground_mock,
        convert_mock,
        trc_export_mock,
        load_meta_mock,
        load_outputs_mock,
    ):
        load_outputs_mock.return_value = [{"frame": "frame_000000.jpg", "outputs": [{}]}]
        load_meta_mock.return_value = {"fps": 30.0}
        process_mock.return_value = np.zeros((1, 70, 3), dtype=np.float32)
        transform_mock.return_value = np.zeros((1, 70, 3), dtype=np.float32)
        ground_info_mock.return_value = {
            "applied_mode": "per_frame_snap",
            "contact_frames": 1,
            "flight_frames": 0,
            "scene_ground_used": False,
            "vertical_mode": "legacy_xz_only",
            "vertical_confident_frames": 0,
            "manual_plane_anchor_active": False,
            "manual_plane_fallback_reason": None,
        }
        contact_data_mock.return_value = {}
        post_ik_meta_mock.return_value = {"available": False}
        scene_ground_mock.return_value = {"available": False, "valid_frames": np.array([False])}
        convert_mock.return_value = (
            np.zeros((1, 2, 3), dtype=np.float32),
            ["Neck", "Hip"],
        )
        pose2sim_mock.return_value = {
            "mot": "output/markers_demo_ik.mot",
            "workspace_root": "output/pose2sim_trial",
            "augmented_trc": "output/pose2sim_trial/pose-3d/markers_demo_LSTM.trc",
            "model": "output/markers_demo_model.osim",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "video_outputs.json"
            json_path.write_text("[]", encoding="utf-8")

            results = run_export_stage(
                json_path=str(json_path),
                output_dir=str(root),
                subject_height=1.75,
                subject_mass=70.0,
                fps=30.0,
                global_translation=False,
                skip_ik=False,
                skip_fbx=True,
                person_idx=0,
                ik_backend="pose2sim_augmented",
                show_header=False,
                project_root=root,
            )

        self.assertEqual(results["mot"], "output/markers_demo_ik.mot")
        self.assertEqual(results["ik_backend"], "pose2sim_augmented")
        self.assertEqual(results["pose2sim_workspace"], "output/pose2sim_trial")
        self.assertEqual(
            results["pose2sim_augmented_trc"],
            "output/pose2sim_trial/pose-3d/markers_demo_LSTM.trc",
        )
        direct_ik_mock.assert_not_called()
        pose2sim_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
