"""Graph export helpers for TRC coordinate and MOT angle artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.opensim_ik import load_mot
from src.trc_exporter import load_trc


def _save_trc_coordinate_graphs(trc_path: Path, coords_dir: Path) -> None:
    trc = load_trc(str(trc_path))
    times = trc["times"]
    markers = trc["markers"]
    marker_names = trc["marker_names"]

    coords_dir.mkdir(parents=True, exist_ok=True)
    for marker_index, marker_name in enumerate(marker_names):
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(times, markers[:, marker_index, 0], label="X")
        axis.plot(times, markers[:, marker_index, 1], label="Y")
        axis.plot(times, markers[:, marker_index, 2], label="Z")
        axis.set_title(marker_name)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(f"Coordinate ({trc['units']})")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(coords_dir / f"{marker_name}.png")
        plt.close(figure)


def _save_mot_angle_graphs(mot_path: Path, angles_dir: Path) -> None:
    mot = load_mot(str(mot_path))
    times = mot["time"]
    coordinates = mot["coordinates"]

    angles_dir.mkdir(parents=True, exist_ok=True)
    for coordinate_name, values in coordinates.items():
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(times, values)
        axis.set_title(coordinate_name)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(angles_dir / f"{coordinate_name}.png")
        plt.close(figure)


def save_export_graphs(
    *,
    trc_path: str | Path,
    mot_path: Optional[str | Path],
    output_dir: str | Path,
) -> dict[str, Optional[str]]:
    """Save coordinate and angle graph images under graphs/."""
    output_dir = Path(output_dir)
    trc_path = Path(trc_path)
    mot_path = None if mot_path is None else Path(mot_path)

    graphs_dir = output_dir / "graphs"
    coords_dir = graphs_dir / "coords"
    angles_dir = graphs_dir / "angles"

    _save_trc_coordinate_graphs(trc_path, coords_dir)
    if mot_path is not None:
        _save_mot_angle_graphs(mot_path, angles_dir)

    return {
        "graphs_dir": str(graphs_dir),
        "coords_dir": str(coords_dir),
        "angles_dir": str(angles_dir) if mot_path is not None else None,
    }
