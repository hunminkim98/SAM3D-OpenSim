import unittest

import run_export
import run_full_pipeline
import run_pipeline


class IkBackendCliTests(unittest.TestCase):
    def test_run_export_accepts_pose2sim_backend(self):
        args = run_export.parse_args(
            ["--input", "video_outputs.json", "--ik-backend", "pose2sim_augmented"]
        )
        self.assertEqual(args.ik_backend, "pose2sim_augmented")

    def test_run_pipeline_accepts_pose2sim_backend(self):
        args = run_pipeline.parse_args(
            ["--input", "video.mp4", "--ik-backend", "pose2sim_augmented"]
        )
        self.assertEqual(args.ik_backend, "pose2sim_augmented")

    def test_run_full_pipeline_accepts_pose2sim_backend(self):
        args = run_full_pipeline.parse_args(
            ["--input", "video.mp4", "--ik-backend", "pose2sim_augmented"]
        )
        self.assertEqual(args.ik_backend, "pose2sim_augmented")


if __name__ == "__main__":
    unittest.main()
