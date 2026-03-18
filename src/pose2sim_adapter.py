"""Config adapter for Pose2Sim marker augmentation and kinematics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def build_pose2sim_config(
    *,
    project_dir: str | Path,
    participant_height: float,
    participant_mass: float,
    fps: float,
    frame_range: str | list[int] = "all",
) -> Dict[str, object]:
    """Build the minimal config dictionary for Pose2Sim augmentation + kinematics."""
    project_dir = str(Path(project_dir))
    return {
        "project": {
            "project_dir": project_dir,
            "multi_person": False,
            "participant_height": float(participant_height),
            "participant_mass": float(participant_mass),
            "frame_rate": float(fps),
            "frame_range": frame_range,
            "exclude_from_batch": [],
        },
        "pose": {
            "pose_model": "Body_with_feet",
        },
        "markerAugmentation": {
            "feet_on_floor": False,
            "make_c3d": False,
        },
        "kinematics": {
            "use_augmentation": True,
            "use_simple_model": False,
            "right_left_symmetry": True,
            "default_height": 1.7,
            "remove_individual_scaling_setup": True,
            "remove_individual_ik_setup": True,
            "fastest_frames_to_remove_percent": 0.1,
            "close_to_zero_speed_m": 0.2,
            "large_hip_knee_angles": 45,
            "trimmed_extrema_percent": 0.5,
        },
        "logging": {
            "use_custom_logging": False,
        },
    }
