import unittest

from src.opensim_marker_spec import build_ik_taskset_xml, get_runtime_ik_marker_specs


class RuntimeIkUpperLimbTests(unittest.TestCase):
    def test_runtime_upper_limb_markers_use_forearm_runtime_bodies(self):
        specs = {
            spec["name"]: spec
            for spec in get_runtime_ik_marker_specs()
        }

        self.assertEqual(specs["LElbow"]["body"], "humerus_l")
        self.assertEqual(specs["RElbow"]["body"], "humerus_r")
        self.assertEqual(specs["LWrist"]["body"], "radius_l")
        self.assertEqual(specs["RWrist"]["body"], "radius_r")
        self.assertEqual(specs["LIndex3"]["body"], "radius_l")
        self.assertEqual(specs["RIndex3"]["body"], "radius_r")
        self.assertEqual(specs["LMiddleTip"]["body"], "radius_l")
        self.assertEqual(specs["RMiddleTip"]["body"], "radius_r")

    def test_runtime_upper_limb_subset_excludes_wrist_dof_recovery_markers(self):
        names = [spec["name"] for spec in get_runtime_ik_marker_specs()]

        self.assertNotIn("LOlecranon", names)
        self.assertNotIn("ROlecranon", names)
        self.assertNotIn("LCubitalFossa", names)
        self.assertNotIn("RCubitalFossa", names)
        self.assertEqual(len(names), 28)

    def test_ik_taskset_xml_includes_hand_markers_without_elbow_cluster_markers(self):
        xml = build_ik_taskset_xml(get_runtime_ik_marker_specs())

        for marker_name in (
            "LWrist",
            "RWrist",
            "LIndex3",
            "RIndex3",
            "LMiddleTip",
            "RMiddleTip",
        ):
            self.assertIn(f'<IKMarkerTask name="{marker_name}">', xml)

        for marker_name in (
            "LOlecranon",
            "ROlecranon",
            "LCubitalFossa",
            "RCubitalFossa",
        ):
            self.assertNotIn(f'<IKMarkerTask name="{marker_name}">', xml)


if __name__ == "__main__":
    unittest.main()
