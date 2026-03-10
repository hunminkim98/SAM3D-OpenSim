"""
OpenSim inverse kinematics solver.

This module provides an interface to run inverse kinematics
using OpenSim/Pose2Sim to compute joint angles from marker data.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.opensim_marker_spec import (
    build_ik_taskset_xml,
    format_lower_body_marker_summary,
    get_runtime_ik_marker_specs,
)
from src.post_ik_foot_snap import apply_post_ik_foot_snap


class OpenSimIK:
    """
    Inverse kinematics solver using OpenSim.

    Supports both pyopensim (direct OpenSim API) and
    Pose2Sim (wrapper with additional features).
    """

    def __init__(
        self,
        model_path: str,
        markers_xml_path: Optional[str] = None,
        accuracy: float = 1e-5,
        max_iterations: int = 1000,
        use_pose2sim: bool = True,
    ):
        """
        Initialize IK solver.

        Args:
            model_path: Path to OpenSim model (.osim)
            markers_xml_path: Path to marker definitions (.xml)
            accuracy: IK solver accuracy
            max_iterations: Maximum IK iterations
            use_pose2sim: Whether to use Pose2Sim (recommended)
        """
        self.model_path = Path(model_path)
        self.markers_xml_path = Path(markers_xml_path) if markers_xml_path else None
        self.accuracy = accuracy
        self.max_iterations = max_iterations
        self.use_pose2sim = use_pose2sim

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def run_ik(
        self,
        trc_path: str,
        output_dir: str,
        subject_height: float = 1.75,
        subject_mass: float = 70.0,
        post_ik_foot_snap_mode: str = "off",
    ) -> Dict[str, Any]:
        """
        Run inverse kinematics on marker data.

        Args:
            trc_path: Path to TRC file with marker trajectories
            output_dir: Output directory for results
            subject_height: Subject height in meters (for scaling)
            subject_mass: Subject mass in kg

        Returns:
            Dictionary with paths to output files:
                - mot: Joint angles (.mot)
                - scaled_model: Scaled model (.osim)
                - ik_setup: IK setup file (.xml)
        """
        trc_path = Path(trc_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_pose2sim:
            return self._run_pose2sim_ik(
                trc_path, output_dir, subject_height, subject_mass, post_ik_foot_snap_mode
            )
        else:
            return self._run_opensim_ik(
                trc_path, output_dir, subject_height, subject_mass, post_ik_foot_snap_mode
            )

    def _run_pose2sim_ik(
        self,
        trc_path: Path,
        output_dir: Path,
        subject_height: float,
        subject_mass: float,
        post_ik_foot_snap_mode: str = "off",
    ) -> Dict[str, Any]:
        """Run IK using Pose2Sim."""
        try:
            from pose2sim import Pose2Sim
            from pose2sim.Utilities import trc_from_mot_osim
        except ImportError:
            print("Pose2Sim not found, falling back to direct OpenSim IK")
            return self._run_opensim_ik(
                trc_path, output_dir, subject_height, subject_mass, post_ik_foot_snap_mode
            )

        # Create Pose2Sim session directory structure
        session_dir = output_dir / "pose2sim_session"
        session_dir.mkdir(exist_ok=True)

        # Copy required files
        model_dest = session_dir / self.model_path.name
        shutil.copy(self.model_path, model_dest)

        trc_dest = session_dir / trc_path.name
        shutil.copy(trc_path, trc_dest)

        if self.markers_xml_path and self.markers_xml_path.exists():
            markers_dest = session_dir / self.markers_xml_path.name
            shutil.copy(self.markers_xml_path, markers_dest)

        # Create config for Pose2Sim
        config = self._create_pose2sim_config(
            session_dir, trc_dest.name, subject_height, subject_mass
        )

        # Run IK
        try:
            Pose2Sim.kinematics(config)

            # Find output files
            mot_files = list(session_dir.glob("*.mot"))
            mot_path = mot_files[0] if mot_files else None

            return {
                "mot": str(mot_path) if mot_path else None,
                "scaled_model": str(model_dest),
                "session_dir": str(session_dir),
            }
        except Exception as e:
            print(f"Pose2Sim IK failed: {e}")
            return self._run_opensim_ik(
                trc_path, output_dir, subject_height, subject_mass, post_ik_foot_snap_mode
            )

    def _run_opensim_ik(
        self,
        trc_path: Path,
        output_dir: Path,
        subject_height: float,
        subject_mass: float,
        post_ik_foot_snap_mode: str = "off",
    ) -> Dict[str, Any]:
        """Run IK using direct OpenSim API."""
        try:
            import opensim as osim
        except ImportError:
            raise ImportError(
                "OpenSim Python bindings not found. "
                "Install with: conda install -c opensim-org opensim"
            )

        # Load model
        model = osim.Model(str(self.model_path))
        model.initSystem()

        # Scale model to subject
        scaled_model_path = output_dir / f"{self.model_path.stem}_scaled.osim"
        self._scale_model(model, subject_height, subject_mass, scaled_model_path)

        # Load scaled model and attach runtime markers.
        scaled_model = osim.Model(str(scaled_model_path))
        scaled_model.initSystem()

        marker_specs = get_runtime_ik_marker_specs(
            markers_xml_path=self.markers_xml_path
        )
        print(f"IK lower-body markers: {format_lower_body_marker_summary(marker_specs)}")
        print(
            "Configured IK foot markers: "
            + ", ".join(
                f'{spec["name"]}={float(spec["weight"]):.2f}'
                for spec in marker_specs
                if spec["name"] in ("LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel")
            )
        )

        for spec in marker_specs:
            marker_name = str(spec["name"])
            body_name = str(spec["body"])
            x, y, z = spec["location"]
            try:
                body = scaled_model.getBodySet().get(body_name)
                marker = osim.Marker()
                marker.setName(marker_name)
                marker.setParentFrame(body)
                marker.set_location(osim.Vec3(float(x), float(y), float(z)))
                scaled_model.addMarker(marker)
            except Exception as exc:
                print(f"Warning: Could not add {marker_name}: {exc}")

        scaled_model.finalizeConnections()
        scaled_model.initSystem()

        model_with_markers_path = output_dir / f"{scaled_model_path.stem}_with_markers.osim"
        scaled_model.printToXML(str(model_with_markers_path))

        marker_tasks_path = output_dir / "ik_markers.xml"
        marker_tasks_path.write_text(
            build_ik_taskset_xml(marker_specs),
            encoding="utf-8",
        )

        # Get time range from TRC
        marker_data = osim.MarkerData(str(trc_path))
        start_time = marker_data.getStartFrameTime()
        end_time = marker_data.getLastFrameTime()

        mot_path = output_dir / f"{trc_path.stem}.mot"
        ik_setup_path = output_dir / "ik_setup.xml"
        ik_setup_xml = '<?xml version="1.0" encoding="UTF-8" ?>\n'
        ik_setup_xml += '<OpenSimDocument Version="40000">\n'
        ik_setup_xml += '    <InverseKinematicsTool name="ik_tool">\n'
        ik_setup_xml += f'        <model_file>{str(model_with_markers_path)}</model_file>\n'
        ik_setup_xml += '        <constraint_weight>20</constraint_weight>\n'
        ik_setup_xml += f'        <accuracy>{self.accuracy}</accuracy>\n'
        ik_setup_xml += f'        <marker_file>{str(trc_path)}</marker_file>\n'
        ik_setup_xml += '        <coordinate_file></coordinate_file>\n'
        ik_setup_xml += f'        <time_range>{start_time} {end_time}</time_range>\n'
        ik_setup_xml += f'        <output_motion_file>{str(mot_path)}</output_motion_file>\n'
        ik_setup_xml += '        <report_errors>true</report_errors>\n'
        ik_setup_xml += '        <report_marker_locations>false</report_marker_locations>\n'
        ik_setup_xml += f'        <results_directory>{str(output_dir)}</results_directory>\n'
        ik_setup_xml += f'        <IKTaskSet file="{str(marker_tasks_path)}"/>\n'
        ik_setup_xml += '    </InverseKinematicsTool>\n'
        ik_setup_xml += '</OpenSimDocument>\n'
        ik_setup_path.write_text(ik_setup_xml, encoding="utf-8")

        print(f"Running IK from {start_time:.3f}s to {end_time:.3f}s...")
        ik_tool = osim.InverseKinematicsTool(str(ik_setup_path))
        ik_tool.run()

        snap_report = None
        if post_ik_foot_snap_mode != "off":
            snap_report = apply_post_ik_foot_snap(
                model_path=model_with_markers_path,
                mot_path=mot_path,
                output_dir=output_dir,
                contact_meta_path=output_dir / "post_ik_contact_meta.json",
                mode=post_ik_foot_snap_mode,
            )
            print(
                "Post-IK foot snap: "
                f"{snap_report.get('status')}, "
                f"corrected_frames={snap_report.get('corrected_frames', 0)}, "
                f"max_drop={snap_report.get('max_applied_drop_m', 0.0):.4f} m"
            )

        return {
            "mot": str(mot_path),
            "scaled_model": str(model_with_markers_path),
            "ik_setup": str(ik_setup_path),
            "post_ik_snap_report": snap_report,
        }

    def _scale_model(
        self,
        model,
        subject_height: float,
        subject_mass: float,
        output_path: Path,
    ) -> None:
        """Scale OpenSim model to subject dimensions."""
        import opensim as osim

        # Simple uniform scaling based on height
        # Assume model is for 1.75m reference height
        reference_height = 1.75
        scale_factor = subject_height / reference_height

        # Create scale tool
        scale_tool = osim.ScaleTool()
        scale_tool.setSubjectMass(subject_mass)

        # Apply uniform scaling to all bodies
        scale_set = osim.ScaleSet()

        for i in range(model.getBodySet().getSize()):
            body = model.getBodySet().get(i)
            body_name = body.getName()

            scale = osim.Scale()
            scale.setSegmentName(body_name)
            scale.setScaleFactors(osim.Vec3(scale_factor, scale_factor, scale_factor))
            scale.setApply(True)
            scale_set.cloneAndAppend(scale)

        # Save scaled model
        model.printToXML(str(output_path))

    def _create_pose2sim_config(
        self,
        session_dir: Path,
        trc_filename: str,
        subject_height: float,
        subject_mass: float,
    ) -> dict:
        """Create Pose2Sim configuration dictionary."""
        return {
            "project": {
                "session_dir": str(session_dir),
            },
            "participant": {
                "height": subject_height,
                "mass": subject_mass,
            },
            "kinematics": {
                "opensim_model": str(self.model_path.name),
                "mode": "ik",
                "use_augmentation": False,
                "right_left_symmetry": True,
                "remove_scaling_setup": True,
            },
        }


def load_mot(mot_path: str) -> Dict[str, np.ndarray]:
    """
    Load motion file (.mot) and return joint angles.

    Args:
        mot_path: Path to MOT file

    Returns:
        Dictionary with:
            - time: (T,) time values
            - coordinates: Dict mapping coordinate names to (T,) values
    """
    with open(mot_path, "r") as f:
        lines = f.readlines()

    # Find header line (starts with "time")
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            header_idx = i
            break

    # Parse header
    header = lines[header_idx].strip().split("\t")
    coord_names = header[1:]  # Skip "time"

    # Parse data
    data_lines = lines[header_idx + 1 :]
    num_frames = len(data_lines)

    time = np.zeros(num_frames)
    coordinates = {name: np.zeros(num_frames) for name in coord_names}

    for i, line in enumerate(data_lines):
        values = line.strip().split("\t")
        time[i] = float(values[0])
        for j, name in enumerate(coord_names):
            coordinates[name][i] = float(values[j + 1])

    return {
        "time": time,
        "coordinates": coordinates,
    }


def save_mot(
    time: np.ndarray,
    coordinates: Dict[str, np.ndarray],
    output_path: str,
    name: str = "Kinematics",
) -> str:
    """
    Save joint angles to MOT file.

    Args:
        time: (T,) time values
        coordinates: Dict mapping coordinate names to (T,) values
        output_path: Output file path
        name: Motion name

    Returns:
        Path to created MOT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coord_names = list(coordinates.keys())
    num_frames = len(time)
    num_coords = len(coord_names)

    lines = []

    # Header
    lines.append(name)
    lines.append("version=1")
    lines.append(f"nRows={num_frames}")
    lines.append(f"nColumns={num_coords + 1}")
    lines.append("inDegrees=yes")
    lines.append("")
    lines.append("Units are S.I. units (second, meters, Newtons, ...)")
    lines.append(
        "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no)."
    )
    lines.append("")
    lines.append("endheader")

    # Column headers
    lines.append("time\t" + "\t".join(coord_names))

    # Data
    for i in range(num_frames):
        line = f"{time[i]:.6f}"
        for name in coord_names:
            line += f"\t{coordinates[name][i]:.6f}"
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return str(output_path)
