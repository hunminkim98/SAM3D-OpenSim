import unittest
from pathlib import Path
from unittest.mock import patch

from src.subprocess_pipeline_runner import run_stage1_subprocess


class SubprocessMeshOutputsTests(unittest.TestCase):
    @patch("src.subprocess_pipeline_runner.subprocess.run")
    def test_run_stage1_subprocess_appends_mesh_flags(self, subprocess_run_mock):
        subprocess_run_mock.return_value.returncode = 0

        root = Path("project")
        input_path = root / "video.mp4"
        output_dir = root / "output"
        with patch.object(Path, "exists", return_value=True):
            run_stage1_subprocess(
                project_root=root,
                input_path=input_path,
                output_dir=output_dir,
                fps=30.0,
                detector="vitdet",
                segmentor=None,
                fov="moge2",
                use_mask=False,
                single_person=True,
                support_surface_mode="auto",
                save_mesh_video=True,
                save_mesh_sequence=True,
                mesh_sequence_format="obj",
                inference_python="python",
            )

        command = subprocess_run_mock.call_args.args[0]
        self.assertIn("--save-mesh-video", command)
        self.assertIn("true", command)
        self.assertIn("--save-mesh-sequence", command)
        self.assertIn("--mesh-sequence-format", command)
        self.assertIn("obj", command)


if __name__ == "__main__":
    unittest.main()
