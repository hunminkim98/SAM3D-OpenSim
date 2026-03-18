import unittest

from src.post_ik_foot_snap import _resolve_marker_name_aliases


class _FakeMarkerSet:
    def __init__(self, names):
        self._names = set(names)

    def get(self, name):
        if name not in self._names:
            raise RuntimeError(name)
        return name


class PostIkFootSnapAliasTests(unittest.TestCase):
    def test_resolve_marker_name_aliases_prefers_direct_names(self):
        marker_set = _FakeMarkerSet(
            ["LHeel", "LBigToe", "LSmallToe", "RHeel", "RBigToe", "RSmallToe"]
        )

        resolved = _resolve_marker_name_aliases(
            marker_set,
            ["LHeel", "LBigToe", "LSmallToe", "RHeel", "RBigToe", "RSmallToe"],
        )

        self.assertEqual(resolved["LHeel"], "LHeel")
        self.assertEqual(resolved["RBigToe"], "RBigToe")

    def test_resolve_marker_name_aliases_supports_pose2sim_lstm_markers(self):
        marker_set = _FakeMarkerSet(
            [
                "L_calc_study",
                "L_toe_study",
                "L_5meta_study",
                "r_calc_study",
                "r_toe_study",
                "r_5meta_study",
            ]
        )

        resolved = _resolve_marker_name_aliases(
            marker_set,
            ["LHeel", "LBigToe", "LSmallToe", "RHeel", "RBigToe", "RSmallToe"],
        )

        self.assertEqual(resolved["LHeel"], "L_calc_study")
        self.assertEqual(resolved["LBigToe"], "L_toe_study")
        self.assertEqual(resolved["LSmallToe"], "L_5meta_study")
        self.assertEqual(resolved["RHeel"], "r_calc_study")
        self.assertEqual(resolved["RBigToe"], "r_toe_study")
        self.assertEqual(resolved["RSmallToe"], "r_5meta_study")

    def test_resolve_marker_name_aliases_raises_when_no_alias_matches(self):
        marker_set = _FakeMarkerSet(["LHeel"])

        with self.assertRaises(ValueError):
            _resolve_marker_name_aliases(
                marker_set,
                ["LHeel", "LBigToe"],
            )


if __name__ == "__main__":
    unittest.main()
