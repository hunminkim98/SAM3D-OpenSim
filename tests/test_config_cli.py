import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_export
import run_full_pipeline
from sam3d_opensim import cli as console_cli
from utils.pipeline_options import load_cli_defaults


class ConfigCliTests(unittest.TestCase):
    def test_run_full_pipeline_parse_args_uses_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake")
            config_path = Path(tmpdir) / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[input]",
                        'video_path = "video.mp4"',
                        "",
                        "[run]",
                        "skip_ik = true",
                        "",
                        "[processing]",
                        'support_surface_mode = "manual_roi"',
                    ]
                ),
                encoding="utf-8",
            )

            args = run_full_pipeline.parse_args(["--config", str(config_path)])

        self.assertEqual(args.input, str(video_path.resolve()))
        self.assertTrue(args.skip_ik)
        self.assertEqual(args.support_surface_mode, "manual_roi")

    def test_run_export_parse_args_uses_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "video_outputs.json"
            json_path.write_text("[]", encoding="utf-8")
            config_path = Path(tmpdir) / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[input]",
                        'video_outputs_path = "video_outputs.json"',
                        "",
                        "[run]",
                        "global_translation = true",
                    ]
                ),
                encoding="utf-8",
            )

            args = run_export.parse_args(["--config", str(config_path)])

        self.assertEqual(args.input, str(json_path.resolve()))
        self.assertTrue(args.global_translation)

    @patch("sam3d_opensim.cli.run_subprocess_pipeline")
    def test_console_cli_runs_full_mode_from_config(self, run_subprocess_pipeline_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            video_path.write_bytes(b"fake")
            config_path = Path(tmpdir) / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'mode = "full"',
                        "",
                        "[input]",
                        'video_path = "video.mp4"',
                    ]
                ),
                encoding="utf-8",
            )

            console_cli.main(["--config", str(config_path)])

        self.assertEqual(
            run_subprocess_pipeline_mock.call_args.kwargs["input_path"],
            str(video_path.resolve()),
        )
        self.assertEqual(
            run_subprocess_pipeline_mock.call_args.kwargs["config_path"],
            str(config_path),
        )
        self.assertEqual(
            run_subprocess_pipeline_mock.call_args.kwargs["workspace_root"],
            Path.cwd(),
        )

    @patch("sam3d_opensim.cli.run_subprocess_pipeline")
    def test_console_cli_allows_skip_inference_without_input_video(self, run_subprocess_pipeline_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "existing_output"
            output_dir.mkdir()
            (output_dir / "video_outputs.json").write_text("[]", encoding="utf-8")
            config_path = Path(tmpdir) / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'mode = "full"',
                        "skip_inference = true",
                        "",
                        "[input]",
                        'video_outputs_path = "existing_output/video_outputs.json"',
                        "",
                        "[output]",
                        'directory = "existing_output"',
                    ]
                ),
                encoding="utf-8",
            )

            console_cli.main(["--config", str(config_path)])

        self.assertIsNone(run_subprocess_pipeline_mock.call_args.kwargs["input_path"])
        self.assertEqual(
            run_subprocess_pipeline_mock.call_args.kwargs["output_dir"],
            str(output_dir.resolve()),
        )

    def test_invalid_run_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "Config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[run]",
                        'mode = "bogus"',
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_cli_defaults(str(config_path))

    def test_missing_config_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_cli_defaults("/tmp/definitely-missing-sam3d-opensim-config.toml")


if __name__ == "__main__":
    unittest.main()
