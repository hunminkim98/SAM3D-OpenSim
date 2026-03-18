"""Pose2Sim marker augmentation + LSTM kinematics runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from src.pose2sim_adapter import build_pose2sim_config
from src.pose2sim_workspace import create_pose2sim_trial_workspace, export_pose2sim_input_trc
from utils.windows_paths import require_active_or_conda_env_python, require_pose2sim_setup


def run_pose2sim_augmented_ik(
    *,
    markers_m: np.ndarray,
    marker_names: Sequence[str],
    fps: float,
    trc_stem: str,
    output_dir: str | Path,
    project_root: str | Path,
    subject_height: float,
    subject_mass: float,
    post_ik_foot_snap_mode: str = "off",
) -> Dict[str, str | None]:
    """Run Pose2Sim marker augmentation followed by LSTM kinematics."""
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    workspace = create_pose2sim_trial_workspace(output_dir)
    input_trc_path = export_pose2sim_input_trc(
        pose_3d_dir=workspace["pose_3d"],
        trc_stem=trc_stem,
        markers=markers_m,
        marker_names=marker_names,
        fps=fps,
    )

    config = build_pose2sim_config(
        project_dir=workspace["root"],
        participant_height=subject_height,
        participant_mass=subject_mass,
        fps=fps,
    )

    opensim_python = require_active_or_conda_env_python(
        "Pose2Sim",
        override_vars=("SAM3D_OPENSIM_OPENSIM_PYTHON", "OPENSIM_PYTHON"),
    )
    require_pose2sim_setup(
        opensim_python=opensim_python,
        override_vars=("SAM3D_OPENSIM_POSE2SIM_SETUP", "POSE2SIM_SETUP"),
    )

    final_mot_path = output_dir / f"{trc_stem}_ik.mot"
    final_model_path = output_dir / f"{trc_stem}_model.osim"
    payload_path = output_dir / "_run_pose2sim_augmented_ik.payload.json"
    payload = {
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "workspace_root": workspace["root"],
        "input_trc_path": str(input_trc_path),
        "final_mot_path": str(final_mot_path),
        "final_model_path": str(final_model_path),
        "post_ik_foot_snap_mode": post_ik_foot_snap_mode,
        "config": config,
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    payload_path_json = json.dumps(str(payload_path))

    script = '''
import json
import shutil
import sys
from pathlib import Path

from Pose2Sim import Pose2Sim

payload_path = Path({payload_path_json})
payload = json.loads(payload_path.read_text(encoding="utf-8"))

project_root = Path(payload["project_root"])
output_dir = Path(payload["output_dir"])
workspace_root = Path(payload["workspace_root"])
input_trc_path = Path(payload["input_trc_path"])
final_mot_path = Path(payload["final_mot_path"])
final_model_path = Path(payload["final_model_path"])
post_ik_foot_snap_mode = payload["post_ik_foot_snap_mode"]
config = payload["config"]

Pose2Sim.markerAugmentation(config)
Pose2Sim.kinematics(config)

pose_3d_dir = workspace_root / "pose-3d"
kinematics_dir = workspace_root / "kinematics"
augmented_trc_path = pose_3d_dir / f"{{input_trc_path.stem}}_LSTM.trc"
if not augmented_trc_path.exists():
    raise FileNotFoundError(f"Expected augmented TRC missing: {{augmented_trc_path}}")

mot_path = kinematics_dir / f"{{augmented_trc_path.stem}}.mot"
model_path = kinematics_dir / f"{{augmented_trc_path.stem}}.osim"
if not mot_path.exists():
    raise FileNotFoundError(f"Expected MOT missing: {{mot_path}}")

shutil.copy2(mot_path, final_mot_path)
if model_path.exists():
    shutil.copy2(model_path, final_model_path)

if post_ik_foot_snap_mode != "off":
    if not final_model_path.exists():
        raise FileNotFoundError(f"Expected model for post-IK foot snap missing: {{final_model_path}}")
    sys.path.insert(0, str(project_root))
    from src.post_ik_foot_snap import apply_post_ik_foot_snap

    apply_post_ik_foot_snap(
        model_path=final_model_path,
        mot_path=final_mot_path,
        output_dir=output_dir,
        contact_meta_path=output_dir / "post_ik_contact_meta.json",
        mode=post_ik_foot_snap_mode,
    )

print(f"SUCCESS_MOT={{final_mot_path}}")
print(f"SUCCESS_AUGMENTED_TRC={{augmented_trc_path}}")
print(f"SUCCESS_MODEL={{final_model_path if final_model_path.exists() else ''}}")
'''.format(
        payload_path_json=payload_path_json,
    )

    script_path = output_dir / "_run_pose2sim_augmented_ik.py"
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [str(opensim_python), str(script_path)],
        cwd=workspace["root"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        for line in result.stderr.split("\n"):
            if line and not line.startswith("[info]"):
                print(line)

    script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError("Pose2Sim augmentation backend failed")
    payload_path.unlink(missing_ok=True)
    if not final_mot_path.exists():
        raise FileNotFoundError(f"Expected final MOT missing: {final_mot_path}")

    augmented_trc_path = Path(workspace["pose_3d"]) / f"{trc_stem}_LSTM.trc"
    return {
        "mot": str(final_mot_path),
        "model": str(final_model_path) if final_model_path.exists() else None,
        "workspace_root": workspace["root"],
        "augmented_trc": str(augmented_trc_path) if augmented_trc_path.exists() else None,
    }
