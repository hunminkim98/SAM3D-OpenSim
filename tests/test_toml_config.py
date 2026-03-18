import tempfile
import unittest
from pathlib import Path

from sam3d_opensim.config import load_pipeline_config


class TomlConfigTests(unittest.TestCase):
    def test_load_pipeline_config_merges_defaults_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "Config.toml"
            input_path = Path(tmpdir) / "video.mp4"
            input_path.write_bytes(b"fake")
            config_path.write_text(
                "\n".join(
                    [
                        "[input]",
                        'video_path = "video.mp4"',
                        "",
                        "[run]",
                        "skip_ik = true",
                        "",
                        "[sam3d]",
                        'device = "cpu"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_pipeline_config(str(config_path))

        self.assertEqual(config["sam3d"]["device"], "cpu")
        self.assertTrue(config["run"]["skip_ik"])
        self.assertEqual(config["input"]["video_path"], str(input_path.resolve()))
        self.assertIn("processing", config)
        self.assertIn("opensim", config)

    def test_windows_style_paths_translate_under_linux(self):
        config = load_pipeline_config("Config.toml")
        self.assertTrue(config["sam3d"]["sam3d_root"])
        if Path("/mnt/c").exists():
            self.assertTrue(
                str(config["sam3d"]["sam3d_root"]).startswith("/mnt/c/"),
                msg=config["sam3d"]["sam3d_root"],
            )


if __name__ == "__main__":
    unittest.main()
