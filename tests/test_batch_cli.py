import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_full_pipeline
import run_inference
from sam3d_opensim import cli as console_cli
from sam3d_opensim.batch import collect_video_inputs, is_batch_input, resolve_video_output_dir


class BatchHelperTests(unittest.TestCase):
    def test_collect_video_inputs_supports_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "b.mp4").write_bytes(b"fake")
            (root / "a.mov").write_bytes(b"fake")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            files = collect_video_inputs(root)
            self.assertTrue(is_batch_input(root))

        self.assertEqual([path.name for path in files], ["a.mov", "b.mp4"])

    def test_resolve_video_output_dir_uses_base_directory_for_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "clip.mp4"
            base_output = Path(tmpdir) / "outputs"
            video_path.write_bytes(b"fake")

            resolved = resolve_video_output_dir(
                video_path=video_path,
                configured_output_dir=base_output,
                batch_mode=True,
            )

        self.assertEqual(resolved.parent, base_output.resolve())
        self.assertIn("clip", resolved.name)


class BatchCliTests(unittest.TestCase):
    @patch("sam3d_opensim.cli.run_subprocess_pipeline")
    def test_console_cli_full_mode_batches_directory_input(self, run_subprocess_pipeline_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_dir = root / "videos"
            output_dir = root / "outputs"
            video_dir.mkdir()
            output_dir.mkdir()
            (video_dir / "a.mp4").write_bytes(b"fake")
            (video_dir / "b.mov").write_bytes(b"fake")
            config_path = root / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'mode = "full"',
                        "",
                        "[input]",
                        'video_path = "videos"',
                        "",
                        "[output]",
                        'directory = "outputs"',
                    ]
                ),
                encoding="utf-8",
            )

            console_cli.main(["--config", str(config_path)])

        self.assertEqual(run_subprocess_pipeline_mock.call_count, 2)
        first_call = run_subprocess_pipeline_mock.call_args_list[0].kwargs
        second_call = run_subprocess_pipeline_mock.call_args_list[1].kwargs
        self.assertTrue(first_call["input_path"].endswith("a.mp4"))
        self.assertTrue(second_call["input_path"].endswith("b.mov"))
        self.assertEqual(Path(first_call["output_dir"]).parent, output_dir.resolve())
        self.assertEqual(Path(second_call["output_dir"]).parent, output_dir.resolve())

    @patch("run_full_pipeline.run_subprocess_pipeline")
    def test_run_full_pipeline_batches_directory_input(self, run_subprocess_pipeline_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_dir = root / "videos"
            output_dir = root / "outputs"
            video_dir.mkdir()
            output_dir.mkdir()
            (video_dir / "first.mp4").write_bytes(b"fake")
            (video_dir / "second.mp4").write_bytes(b"fake")

            with patch(
                "sys.argv",
                [
                    "run_full_pipeline.py",
                    "--input",
                    str(video_dir),
                    "--output",
                    str(output_dir),
                ],
            ):
                run_full_pipeline.main()

        self.assertEqual(run_subprocess_pipeline_mock.call_count, 2)
        self.assertTrue(
            run_subprocess_pipeline_mock.call_args_list[0].kwargs["input_path"].endswith("first.mp4")
        )
        self.assertTrue(
            run_subprocess_pipeline_mock.call_args_list[1].kwargs["input_path"].endswith("second.mp4")
        )

    @patch("run_inference.run_inference")
    def test_run_inference_batches_directory_input(self, run_inference_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_dir = root / "videos"
            output_dir = root / "outputs"
            video_dir.mkdir()
            output_dir.mkdir()
            (video_dir / "first.mp4").write_bytes(b"fake")
            (video_dir / "second.mov").write_bytes(b"fake")

            with patch(
                "sys.argv",
                [
                    "run_inference.py",
                    "--input",
                    str(video_dir),
                    "--output",
                    str(output_dir),
                ],
            ):
                run_inference.main()

        self.assertEqual(run_inference_mock.call_count, 2)
        self.assertTrue(
            run_inference_mock.call_args_list[0].kwargs["input_path"].endswith("first.mp4")
        )
        self.assertTrue(
            run_inference_mock.call_args_list[1].kwargs["input_path"].endswith("second.mov")
        )


if __name__ == "__main__":
    unittest.main()
