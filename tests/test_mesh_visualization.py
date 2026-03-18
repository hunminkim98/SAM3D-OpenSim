import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.mesh_visualization import (
    _resolve_frame_outputs,
    _require_local_renderer,
    save_mesh_sequence,
    save_mesh_sidecars,
)
from src.sam3d_mesh_renderer import Renderer as LocalMeshRenderer
from src.sam3d_mesh_renderer import _apply_platform_override_from_env


class _FakeMesh:
    def __init__(self):
        self.exports = []

    def export(self, path):
        self.exports.append(str(path))


class MeshVisualizationTests(unittest.TestCase):
    def test_require_local_renderer_returns_repo_local_renderer(self):
        self.assertIs(_require_local_renderer(), LocalMeshRenderer)

    def test_apply_platform_override_from_env_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            _apply_platform_override_from_env()
            self.assertNotIn("PYOPENGL_PLATFORM", os.environ)

        with patch.dict(
            os.environ,
            {"SAM3D_OPENSIM_MESH_RENDER_PLATFORM": "test_backend"},
            clear=True,
        ):
            _apply_platform_override_from_env()
            self.assertEqual(os.environ["PYOPENGL_PLATFORM"], "test_backend")

    def test_resolve_frame_outputs_uses_selected_output_for_single_person(self):
        frame_data = {
            "output": {"id": "selected"},
            "outputs": [{"id": "a"}, {"id": "b"}],
        }
        outputs = _resolve_frame_outputs(frame_data, single_person=True)
        self.assertEqual(outputs, [{"id": "selected"}])

    def test_resolve_frame_outputs_uses_all_outputs_for_multi_person(self):
        frame_data = {
            "output": {"id": "selected"},
            "outputs": [{"id": "a"}, None, {"id": "b"}],
        }
        outputs = _resolve_frame_outputs(frame_data, single_person=False)
        self.assertEqual(outputs, [{"id": "a"}, {"id": "b"}])

    @patch("src.mesh_visualization._build_export_mesh")
    def test_save_mesh_sequence_writes_expected_filenames(self, build_export_mesh_mock):
        fake_mesh_a = _FakeMesh()
        fake_mesh_b = _FakeMesh()
        build_export_mesh_mock.side_effect = [fake_mesh_a, fake_mesh_b]

        frame_outputs = [
            {
                "output": {
                    "pred_vertices": [[0.0, 0.0, 0.0]],
                    "pred_cam_t": [0.0, 0.0, 1.0],
                }
            },
            {
                "output": {
                    "pred_vertices": [[1.0, 0.0, 0.0]],
                    "pred_cam_t": [0.0, 0.0, 2.0],
                }
            },
        ]

        result = save_mesh_sequence(
            frame_outputs=frame_outputs,
            faces=[[0, 0, 0]],
            output_dir=".",
            single_person=True,
            export_format="ply",
        )

        self.assertEqual(result["count"], 2)
        self.assertEqual(
            fake_mesh_a.exports,
            [str(Path("frame_000000_person_000.ply"))],
        )
        self.assertEqual(
            fake_mesh_b.exports,
            [str(Path("frame_000001_person_000.ply"))],
        )

    @patch("src.mesh_visualization.save_mesh_overlay_video")
    @patch("src.mesh_visualization.save_mesh_sequence")
    def test_save_mesh_sidecars_returns_requested_paths(
        self,
        save_mesh_sequence_mock,
        save_mesh_overlay_video_mock,
    ):
        save_mesh_overlay_video_mock.return_value = "output/mesh_vis/overlay.mp4"
        save_mesh_sequence_mock.return_value = {
            "directory": "output/mesh_export",
            "format": "ply",
            "count": 3,
            "files": [],
        }

        result = save_mesh_sidecars(
            output_dir="output",
            frame_paths=["frame_000000.jpg"],
            frame_outputs=[{"output": None}],
            faces=[[0, 0, 0]],
            fps=30.0,
            single_person=True,
            save_mesh_video=True,
            save_mesh_sequence_files=True,
            mesh_sequence_format="ply",
        )

        self.assertEqual(result["mesh_video"], "output/mesh_vis/overlay.mp4")
        self.assertEqual(result["mesh_sequence_dir"], "output/mesh_export")
        self.assertEqual(result["mesh_sequence_format"], "ply")
        self.assertEqual(result["mesh_sequence_count"], 3)

    @patch("src.mesh_visualization.save_mesh_overlay_video")
    def test_save_mesh_sidecars_keeps_pipeline_alive_when_video_fails(
        self,
        save_mesh_overlay_video_mock,
    ):
        save_mesh_overlay_video_mock.side_effect = RuntimeError("renderer failed")

        result = save_mesh_sidecars(
            output_dir="output",
            frame_paths=["frame_000000.jpg"],
            frame_outputs=[{"output": None}],
            faces=[[0, 0, 0]],
            fps=30.0,
            single_person=True,
            save_mesh_video=True,
            save_mesh_sequence_files=False,
            mesh_sequence_format="ply",
        )

        self.assertNotIn("mesh_video", result)
        self.assertEqual(result["mesh_video_error"], "renderer failed")


if __name__ == "__main__":
    unittest.main()
