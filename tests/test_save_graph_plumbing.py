import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_export
import run_full_pipeline
import run_pipeline


class SaveGraphPlumbingTests(unittest.TestCase):
    @patch("run_export.run_export_stage")
    def test_run_export_forwards_save_graph(self, run_export_stage_mock):
        run_export_stage_mock.return_value = {}

        run_export.run_export(
            json_path="video_outputs.json",
            output_dir="output",
            subject_height=1.75,
            subject_mass=70.0,
            fps=30.0,
            global_translation=False,
            skip_ik=True,
            skip_fbx=True,
            person_idx=0,
            save_graph=True,
        )

        self.assertTrue(run_export_stage_mock.call_args.kwargs["save_graph"])

    @patch("run_pipeline.run_combined_pipeline")
    def test_run_pipeline_forwards_save_graph(self, run_combined_pipeline_mock):
        run_combined_pipeline_mock.return_value = {"success": True, "outputs": {}, "timings": {}}

        run_pipeline.run_pipeline(
            input_path="video.mp4",
            save_graph=True,
        )

        self.assertTrue(run_combined_pipeline_mock.call_args.kwargs["save_graph"])

    @patch("run_full_pipeline.run_subprocess_pipeline")
    def test_run_full_pipeline_main_forwards_save_graph(self, run_subprocess_pipeline_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake")

            with patch("sys.argv", ["run_full_pipeline.py", "--input", str(video_path), "--save_graph"]):
                run_full_pipeline.main()

        self.assertTrue(run_subprocess_pipeline_mock.call_args.kwargs["save_graph"])


if __name__ == "__main__":
    unittest.main()
