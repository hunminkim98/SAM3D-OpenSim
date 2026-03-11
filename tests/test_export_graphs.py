import tempfile
import unittest
from pathlib import Path


class ExportGraphsTests(unittest.TestCase):
    def _write_sample_trc(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "PathFileType\t4\t(X/Y/Z)\tsample.trc",
                    "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
                    "30.000000\t30.000000\t2\t1\tmm\t30.000000\t1\t2",
                    "Frame#\tTime\tPelvisCenter\t\t",
                    "\t\tX1\tY1\tZ1",
                    "",
                    "1\t0.000000\t1.0\t2.0\t3.0",
                    "2\t0.033333\t4.0\t5.0\t6.0",
                ]
            ),
            encoding="utf-8",
        )

    def _write_sample_mot(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "Coordinates",
                    "version=1",
                    "nRows=2",
                    "nColumns=3",
                    "inDegrees=yes",
                    "",
                    "Units are S.I. units (second, meters, Newtons, ...)",
                    "",
                    "endheader",
                    "time\tpelvis_tilt\tknee_angle_r",
                    "0.000000\t10.0\t20.0",
                    "0.033333\t15.0\t25.0",
                ]
            ),
            encoding="utf-8",
        )

    def test_save_export_graphs_creates_coord_and_angle_images(self):
        from src.export_graphs import save_export_graphs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trc_path = root / "markers_sample.trc"
            mot_path = root / "markers_sample_ik.mot"
            output_dir = root / "output"
            self._write_sample_trc(trc_path)
            self._write_sample_mot(mot_path)

            result = save_export_graphs(
                trc_path=trc_path,
                mot_path=mot_path,
                output_dir=output_dir,
            )

            coords_dir = output_dir / "graphs" / "coords"
            angles_dir = output_dir / "graphs" / "angles"
            self.assertEqual(result["coords_dir"], str(coords_dir))
            self.assertEqual(result["angles_dir"], str(angles_dir))
            self.assertTrue((coords_dir / "PelvisCenter.png").exists())
            self.assertTrue((angles_dir / "pelvis_tilt.png").exists())
            self.assertTrue((angles_dir / "knee_angle_r.png").exists())


if __name__ == "__main__":
    unittest.main()
