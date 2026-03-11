import unittest

import run_export
import run_full_pipeline
import run_pipeline


class SaveGraphCliTests(unittest.TestCase):
    def test_run_export_accepts_save_graph_flag(self):
        args = run_export.parse_args(
            ["--input", "video_outputs.json", "--save_graph"]
        )
        self.assertTrue(args.save_graph)

    def test_run_pipeline_accepts_save_graph_flag(self):
        args = run_pipeline.parse_args(
            ["--input", "video.mp4", "--save_graph"]
        )
        self.assertTrue(args.save_graph)

    def test_run_full_pipeline_accepts_save_graph_flag(self):
        args = run_full_pipeline.parse_args(
            ["--input", "video.mp4", "--save_graph"]
        )
        self.assertTrue(args.save_graph)


if __name__ == "__main__":
    unittest.main()
