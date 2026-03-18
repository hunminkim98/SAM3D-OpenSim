import unittest

import run_full_pipeline
import run_inference
import run_pipeline


class MeshCliTests(unittest.TestCase):
    def test_run_inference_accepts_mesh_flags(self):
        args = run_inference.parse_args(
            [
                "--input",
                "video.mp4",
                "--save-mesh-video",
                "--save-mesh-sequence",
                "--mesh-sequence-format",
                "obj",
            ]
        )
        self.assertTrue(args.save_mesh_video)
        self.assertTrue(args.save_mesh_sequence)
        self.assertEqual(args.mesh_sequence_format, "obj")

    def test_run_pipeline_accepts_mesh_flags(self):
        args = run_pipeline.parse_args(
            [
                "--input",
                "video.mp4",
                "--save-mesh-video",
                "--save-mesh-sequence",
                "--mesh-sequence-format",
                "obj",
            ]
        )
        self.assertTrue(args.save_mesh_video)
        self.assertTrue(args.save_mesh_sequence)
        self.assertEqual(args.mesh_sequence_format, "obj")

    def test_run_full_pipeline_accepts_mesh_flags(self):
        args = run_full_pipeline.parse_args(
            [
                "--input",
                "video.mp4",
                "--save-mesh-video",
                "--save-mesh-sequence",
                "--mesh-sequence-format",
                "obj",
            ]
        )
        self.assertTrue(args.save_mesh_video)
        self.assertTrue(args.save_mesh_sequence)
        self.assertEqual(args.mesh_sequence_format, "obj")


if __name__ == "__main__":
    unittest.main()
