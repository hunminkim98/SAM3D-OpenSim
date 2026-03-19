import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.export_stage import run_export_stage


class ExportStageSaveGraphTests(unittest.TestCase):
    @patch("src.export_stage.load_video_outputs")
    @patch("src.export_stage.load_inference_meta")
    @patch("src.export_graphs.save_export_graphs")
    @patch("src.trc_exporter.TRCExporter.export")
    @patch("src.keypoint_converter.KeypointConverter.convert")
    @patch("src.moge_scene_ground.extract_scene_ground_arrays_from_json")
    @patch("src.post_ik_foot_snap.build_post_ik_contact_meta")
    @patch("src.coordinate_transform.CoordinateTransformer.get_last_contact_data")
    @patch("src.coordinate_transform.CoordinateTransformer.get_last_ground_alignment_info")
    @patch("src.coordinate_transform.CoordinateTransformer.transform")
    @patch("src.post_processing.PostProcessor.process")
    def test_run_export_stage_calls_graph_export_for_trc_only(
        self,
        process_mock,
        transform_mock,
        ground_info_mock,
        contact_data_mock,
        post_ik_meta_mock,
        scene_ground_mock,
        convert_mock,
        trc_export_mock,
        save_graphs_mock,
        load_meta_mock,
        load_outputs_mock,
    ):
        load_outputs_mock.return_value = [
            {
                "frame": "frame_000000.jpg",
                "outputs": [
                    {
                        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32).tolist(),
                        "pred_cam_t": [0.0, 0.0, 5.0],
                    }
                ],
            }
        ]
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
            np.zeros((1, 1, 3), dtype=np.float32),
            ["PelvisCenter"],
        )
        save_graphs_mock.return_value = {
            "coords_dir": None,
            "angles_dir": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "video_outputs.json"
            json_path.write_text("[]", encoding="utf-8")
            save_graphs_mock.return_value = {
                "coords_dir": str(root / "graphs" / "coords"),
                "angles_dir": None,
            }

            results = run_export_stage(
                json_path=str(json_path),
                output_dir=str(root),
                subject_height=1.75,
                subject_mass=70.0,
                fps=30.0,
                global_translation=False,
                skip_ik=True,
                skip_fbx=True,
                person_idx=0,
                save_graph=True,
                show_header=False,
                project_root=root,
            )

            self.assertEqual(results["graph_coords_dir"], str(root / "graphs" / "coords"))
            self.assertIsNone(results["graph_angles_dir"])
            save_graphs_mock.assert_called_once()
            self.assertEqual(save_graphs_mock.call_args.kwargs["mot_path"], None)


if __name__ == "__main__":
    unittest.main()
