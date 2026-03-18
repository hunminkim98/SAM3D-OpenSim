import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pose2sim_adapter import build_pose2sim_config
from src.pose2sim_workspace import (
    create_pose2sim_trial_workspace,
    export_pose2sim_input_trc,
)
from utils.pipeline_options import load_cli_defaults


class Pose2SimWorkspaceTests(unittest.TestCase):
    def test_create_pose2sim_trial_workspace_creates_expected_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = create_pose2sim_trial_workspace(tmpdir)

            self.assertTrue(Path(workspace["root"]).is_dir())
            self.assertTrue(Path(workspace["pose_3d"]).is_dir())
            self.assertTrue(Path(workspace["kinematics"]).is_dir())

    def test_create_pose2sim_trial_workspace_cleans_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_pose_3d = Path(tmpdir) / "pose2sim_trial" / "pose-3d"
            stale_kinematics = Path(tmpdir) / "pose2sim_trial" / "kinematics"
            stale_pose_3d.mkdir(parents=True)
            stale_kinematics.mkdir(parents=True)
            (stale_pose_3d / "old_file.trc").write_text("stale", encoding="utf-8")
            (stale_kinematics / "old_file.mot").write_text("stale", encoding="utf-8")

            workspace = create_pose2sim_trial_workspace(tmpdir)

            self.assertFalse((Path(workspace["pose_3d"]) / "old_file.trc").exists())
            self.assertFalse((Path(workspace["kinematics"]) / "old_file.mot").exists())

    def test_export_pose2sim_input_trc_writes_meter_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = create_pose2sim_trial_workspace(tmpdir)
            trc_path = export_pose2sim_input_trc(
                pose_3d_dir=workspace["pose_3d"],
                trc_stem="markers_demo",
                markers=np.zeros((2, 2, 3), dtype=np.float32),
                marker_names=["Neck", "Hip"],
                fps=30.0,
            )

            header_values = Path(trc_path).read_text(encoding="utf-8").splitlines()[2]
            self.assertIn("\tm\t", header_values)

    def test_build_pose2sim_config_emits_required_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_pose2sim_config(
                project_dir=tmpdir,
                participant_height=1.75,
                participant_mass=70.0,
                fps=30.0,
            )

            self.assertEqual(config["project"]["project_dir"], tmpdir)
            self.assertEqual(config["project"]["participant_height"], 1.75)
            self.assertEqual(config["project"]["participant_mass"], 70.0)
            self.assertEqual(config["pose"]["pose_model"], "Body_with_feet")
            self.assertTrue(config["kinematics"]["use_augmentation"])
            self.assertFalse(config["markerAugmentation"]["feet_on_floor"])

    def test_load_cli_defaults_rejects_invalid_ik_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "opensim:\n  ik_backend: not_a_backend\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_cli_defaults(str(config_path))


if __name__ == "__main__":
    unittest.main()
