"""
Shared Blender export helpers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from utils.windows_paths import resolve_blender_executable


def run_blender_fbx_export(
    *,
    mot_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> Optional[Path]:
    """Export a MOT file to FBX using the shared Blender script."""
    mot_path = Path(mot_path).resolve()
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()

    blend_template = project_root / "Import_OS4_Patreon_Aitor_Skely.blend"
    blender_script = project_root / "scripts" / "export_fbx_skely.py"
    blender_path = resolve_blender_executable(
        override_vars=("SAM3D_OPENSIM_BLENDER_PATH", "BLENDER_PATH"),
    )

    if not blender_path:
        print("  Blender not found. Set BLENDER_PATH or install Blender under Program Files.")
        return None

    if not blend_template.exists():
        print(f"  Skeleton template not found: {blend_template}")
        return None

    fbx_path = output_dir / f"{mot_path.stem.replace('_ik', '')}.fbx"
    cmd = [
        str(blender_path),
        "--background",
        str(blend_template),
        "--python",
        str(blender_script),
        "--",
        "--mot",
        str(mot_path),
        "--output",
        str(fbx_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return fbx_path

    if result.stderr:
        for line in result.stderr.split("\n"):
            if line and "Error" in line:
                print(f"  {line}")
    return None
