import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.subprocess_pipeline_runner import run_stage2_subprocess


class SubprocessIkBackendTests(unittest.TestCase):
    @patch("src.subprocess_pipeline_runner.subprocess.run")
    def test_run_stage2_subprocess_appends_ik_backend_flag(self, subprocess_run_mock):
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
                subject_height=1.75,
                subject_mass=70.0,
                smooth=6.0,
                ground_alignment_mode="auto",
                vertical_translation_mode="auto",
                post_ik_foot_snap_mode="off",
                ik_backend="pose2sim_augmented",
                save_graph=False,
                global_translation=False,
                skip_ik=True,
                skip_fbx=True,
            )

        command = subprocess_run_mock.call_args.args[0]
        self.assertIn("--ik-backend", command)
        self.assertIn("pose2sim_augmented", command)


if __name__ == "__main__":
    unittest.main()
