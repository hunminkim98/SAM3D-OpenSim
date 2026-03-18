"""Helpers for creating a Pose2Sim-compatible trial workspace."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from src.trc_exporter import TRCExporter


def create_pose2sim_trial_workspace(output_dir: str | Path) -> Dict[str, str]:
    """Create the deterministic Pose2Sim trial workspace inside an output dir."""
    root = Path(output_dir) / "pose2sim_trial"
    if root.exists():
        shutil.rmtree(root)
    pose_3d_dir = root / "pose-3d"
    kinematics_dir = root / "kinematics"

    pose_3d_dir.mkdir(parents=True, exist_ok=True)
    kinematics_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": str(root),
        "pose_3d": str(pose_3d_dir),
        "kinematics": str(kinematics_dir),
    }


def export_pose2sim_input_trc(
    *,
    pose_3d_dir: str | Path,
    trc_stem: str,
    markers: np.ndarray,
    marker_names: Sequence[str],
    fps: float,
) -> str:
    """Export a Pose2Sim-compatible meter TRC into the trial workspace."""
    pose_3d_dir = Path(pose_3d_dir)
    trc_path = pose_3d_dir / f"{trc_stem}.trc"
    exporter = TRCExporter(fps=fps, units="m")
    exporter.export(markers, list(marker_names), str(trc_path))
    return str(trc_path)
