import unittest
from pathlib import Path
from unittest.mock import patch

import run_full_pipeline
import run_inference
import run_pipeline


class MeshPlumbingTests(unittest.TestCase):
    @patch("run_inference.run_inference_stage")
    def test_run_inference_forwards_mesh_flags(self, run_inference_stage_mock):
        run_inference_stage_mock.return_value = {
            "inference_meta": {"inference_time": 0.0},
            "frame_paths": [],
            "saved_paths": {
                "video_outputs": "video_outputs.json",
                "mesh_video": None,
                "mesh_sequence_dir": None,
            },
        }

        run_inference.run_inference(
            input_path="video.mp4",
            output_dir="output",
            fps=30.0,
            device="cpu",
            save_mesh_video=True,
            save_mesh_sequence=True,
            mesh_sequence_format="obj",
        )

        self.assertTrue(run_inference_stage_mock.call_args.kwargs["save_mesh_video"])
        self.assertTrue(run_inference_stage_mock.call_args.kwargs["save_mesh_sequence"])
        self.assertEqual(
            run_inference_stage_mock.call_args.kwargs["mesh_sequence_format"],
            "obj",
        )

    @patch("run_pipeline.run_combined_pipeline")
    def test_run_pipeline_forwards_mesh_flags(self, run_combined_pipeline_mock):
        run_combined_pipeline_mock.return_value = {
            "success": True,
            "outputs": {},
            "timings": {},
        }

        run_pipeline.run_pipeline(
            input_path="video.mp4",
            save_mesh_video=True,
            save_mesh_sequence=True,
            mesh_sequence_format="obj",
        )

        self.assertTrue(run_combined_pipeline_mock.call_args.kwargs["save_mesh_video"])
        self.assertTrue(run_combined_pipeline_mock.call_args.kwargs["save_mesh_sequence"])
        self.assertEqual(
            run_combined_pipeline_mock.call_args.kwargs["mesh_sequence_format"],
            "obj",
        )

    @patch("run_full_pipeline.run_subprocess_pipeline")
    def test_run_full_pipeline_main_forwards_mesh_flags(self, run_subprocess_pipeline_mock):
        video_path = Path("video.mp4")

        with patch.object(Path, "exists", return_value=True):
            with patch(
                "sys.argv",
                [
                    "run_full_pipeline.py",
                    "--input",
                    str(video_path),
                    "--save-mesh-video",
                    "--save-mesh-sequence",
                    "--mesh-sequence-format",
                    "obj",
                ],
            ):
                run_full_pipeline.main()

        self.assertTrue(run_subprocess_pipeline_mock.call_args.kwargs["save_mesh_video"])
        self.assertTrue(run_subprocess_pipeline_mock.call_args.kwargs["save_mesh_sequence"])
        self.assertEqual(
            run_subprocess_pipeline_mock.call_args.kwargs["mesh_sequence_format"],
            "obj",
        )


if __name__ == "__main__":
    unittest.main()
