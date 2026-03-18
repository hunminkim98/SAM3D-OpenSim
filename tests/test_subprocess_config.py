import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.subprocess_pipeline_runner import run_stage1_subprocess, run_stage2_subprocess


class SubprocessConfigTests(unittest.TestCase):
    @patch("src.subprocess_pipeline_runner.subprocess.run")
    def test_stage1_uses_module_invocation_and_forwards_config(self, subprocess_run_mock):
        subprocess_run_mock.return_value.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "video.mp4"
            output_dir = root / "output"
            input_path.write_bytes(b"fake")
            output_dir.mkdir()
            (output_dir / "video_outputs.json").write_text("[]", encoding="utf-8")

            run_stage1_subprocess(
                project_root=root,
                input_path=input_path,
                output_dir=output_dir,
                config_path=str(root / "Config.toml"),
                fps=30.0,
                detector="vitdet",
                segmentor=None,
                fov="moge2",
                use_mask=False,
                single_person=True,
                support_surface_mode="auto",
                save_mesh_video=False,
                save_mesh_sequence=False,
                mesh_sequence_format="ply",
                inference_python="python",
            )

        command = subprocess_run_mock.call_args.args[0]
        self.assertEqual(command[:3], ["python", "-m", "run_inference"])
        self.assertIn("--config", command)
        self.assertIn("--use-mask", command)
        self.assertIn("false", command)

    @patch("src.subprocess_pipeline_runner.subprocess.run")
    def test_stage2_uses_module_invocation_and_forwards_config(self, subprocess_run_mock):
        subprocess_run_mock.return_value.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "video_outputs.json"
            output_dir = root / "output"
            output_dir.mkdir()
            json_path.write_text("[]", encoding="utf-8")

            run_stage2_subprocess(
                project_root=root,
                current_python="python",
                json_path=json_path,
                output_dir=output_dir,
                config_path=str(root / "Config.toml"),
                subject_height=1.75,
                subject_mass=70.0,
                smooth=6.0,
                ground_alignment_mode="auto",
                vertical_translation_mode="auto",
                post_ik_foot_snap_mode="off",
                save_graph=False,
                global_translation=False,
                skip_ik=True,
                skip_fbx=True,
            )

        command = subprocess_run_mock.call_args.args[0]
        self.assertEqual(command[:3], ["python", "-m", "run_export"])
        self.assertIn("--config", command)
        self.assertIn("--skip-ik", command)
        self.assertIn("--skip-fbx", command)
        self.assertIn("--global-translation", command)
        self.assertEqual(command[command.index("--save_graph") + 1], "false")
        self.assertEqual(command[command.index("--global-translation") + 1], "false")
        self.assertEqual(command[command.index("--skip-ik") + 1], "true")
        self.assertEqual(command[command.index("--skip-fbx") + 1], "true")


if __name__ == "__main__":
    unittest.main()
