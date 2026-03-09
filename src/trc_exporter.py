"""
TRC file exporter for OpenSim.

This module generates Track Row Column (.trc) files compatible
with OpenSim for inverse kinematics analysis.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np


class TRCExporter:
    """
    Exports marker data to TRC format for OpenSim.

    TRC files contain marker trajectory data with timestamps.
    """

    def __init__(
        self,
        fps: float = 30.0,
        units: str = "mm",
    ):
        """
        Initialize TRC exporter.

        Args:
            fps: Frame rate in Hz
            units: Coordinate units ('mm' or 'm')
        """
        self.fps = fps
        self.units = units

    def export(
        self,
        markers: np.ndarray,
        marker_names: List[str],
        output_path: str,
        start_frame: int = 1,
    ) -> str:
        """
        Export marker data to TRC file.

        Args:
            markers: (T, M, 3) array of marker positions
            marker_names: List of M marker names
            output_path: Output file path
            start_frame: Starting frame number (default 1)

        Returns:
            Path to created TRC file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_frames = markers.shape[0]
        num_markers = markers.shape[1]

        if len(marker_names) != num_markers:
            raise ValueError(
                f"Number of marker names ({len(marker_names)}) "
                f"doesn't match data ({num_markers})"
            )

        # Generate TRC content
        lines = []

        # Line 1: Path/file type
        lines.append("PathFileType\t4\t(X/Y/Z)\t" + output_path.name)

        # Line 2: Header info
        lines.append(
            f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
        )

        # Line 3: Header values
        lines.append(
            f"{self.fps:.6f}\t{self.fps:.6f}\t{num_frames}\t{num_markers}\t{self.units}\t{self.fps:.6f}\t{start_frame}\t{num_frames}"
        )

        # Line 4: Marker names (each name appears 3 times for X, Y, Z)
        marker_header = "Frame#\tTime"
        for name in marker_names:
            marker_header += f"\t{name}\t\t"
        lines.append(marker_header.rstrip("\t"))

        # Line 5: Coordinate labels
        coord_header = "\t"
        for i, _ in enumerate(marker_names):
            coord_header += f"\tX{i+1}\tY{i+1}\tZ{i+1}"
        lines.append(coord_header)

        # Line 6: Empty line
        lines.append("")

        # Data lines
        for frame_idx in range(num_frames):
            frame_num = start_frame + frame_idx
            time = frame_idx / self.fps

            # Start with frame number and time
            line = f"{frame_num}\t{time:.6f}"

            # Add marker coordinates
            for marker_idx in range(num_markers):
                x, y, z = markers[frame_idx, marker_idx]
                line += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"

            lines.append(line)

        # Write file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return str(output_path)

    def export_from_keypoints(
        self,
        keypoints_3d: np.ndarray,
        converter,
        output_path: str,
        include_derived: bool = True,
    ) -> str:
        """
        Export directly from MHR70 keypoints using a converter.

        Args:
            keypoints_3d: (T, 70, 3) MHR70 keypoints
            converter: KeypointConverter instance
            output_path: Output file path
            include_derived: Whether to include derived markers

        Returns:
            Path to created TRC file
        """
        # Convert keypoints to markers
        markers, marker_names = converter.convert(
            keypoints_3d, include_derived=include_derived
        )

        return self.export(markers, marker_names, output_path)


def create_trc_from_dict(
    marker_dict: dict,
    output_path: str,
    fps: float = 30.0,
    units: str = "mm",
) -> str:
    """
    Create TRC file from a dictionary of marker trajectories.

    Args:
        marker_dict: Dictionary mapping marker names to (T, 3) arrays
        output_path: Output file path
        fps: Frame rate
        units: Coordinate units

    Returns:
        Path to created TRC file
    """
    marker_names = list(marker_dict.keys())
    num_frames = next(iter(marker_dict.values())).shape[0]

    # Stack markers into array
    markers = np.stack([marker_dict[name] for name in marker_names], axis=1)

    exporter = TRCExporter(fps=fps, units=units)
    return exporter.export(markers, marker_names, output_path)


def load_trc(trc_path: str) -> dict:
    """
    Load a TRC file and return marker data.

    Args:
        trc_path: Path to TRC file

    Returns:
        Dictionary with:
            - markers: (T, M, 3) marker positions
            - marker_names: List of marker names
            - fps: Frame rate
            - units: Coordinate units
            - times: (T,) time values
    """
    with open(trc_path, "r") as f:
        lines = f.readlines()

    # Parse header (line 3)
    header_values = lines[2].strip().split("\t")
    fps = float(header_values[0])
    num_frames = int(header_values[2])
    num_markers = int(header_values[3])
    units = header_values[4]

    # Parse marker names (line 4)
    marker_line = lines[3].strip().split("\t")
    # Skip "Frame#" and "Time", then take every 3rd name (since names repeat for X,Y,Z)
    marker_names = []
    for i in range(2, len(marker_line), 3):
        if marker_line[i].strip():
            marker_names.append(marker_line[i].strip())

    # Parse data (lines 7+)
    markers = np.zeros((num_frames, num_markers, 3))
    times = np.zeros(num_frames)

    for i, line in enumerate(lines[6:]):
        if not line.strip():
            continue

        values = line.strip().split("\t")
        times[i] = float(values[1])

        for j in range(num_markers):
            idx = 2 + j * 3
            markers[i, j, 0] = float(values[idx])
            markers[i, j, 1] = float(values[idx + 1])
            markers[i, j, 2] = float(values[idx + 2])

    return {
        "markers": markers,
        "marker_names": marker_names,
        "fps": fps,
        "units": units,
        "times": times,
    }
