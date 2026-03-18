"""Shared subprocess orchestration for the canonical two-stage pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from src.pipeline_runtime_common import (
    print_pipeline_banner,
    print_pipeline_outputs_summary,
    resolve_pipeline_output_dir,
)
from utils.windows_paths import require_conda_env_python


def _append_optional_arg(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _append_bool_arg(cmd: list[str], flag: str, value: bool) -> None:
    cmd.extend([flag, str(bool(value)).lower()])


def _discover_output_paths(output_dir: Path) -> dict[str, Optional[Path]]:
    return {
        "trc": next(iter(sorted(output_dir.glob("*.trc"))), None),
        "mot": next(iter(sorted(output_dir.glob("*_ik.mot"))), None),
        "fbx": next(iter(sorted(output_dir.glob("*.fbx"))), None),
    }


def run_stage1_subprocess(
    *,
    project_root: Path,
    input_path: Path,
    output_dir: Path,
    config_path: Optional[str] = None,
    fps: float,
    detector: str,
    segmentor: Optional[str],
    fov: str,
    use_mask: bool,
    single_person: bool,
    support_surface_mode: str,
    save_mesh_video: bool,
    save_mesh_sequence: bool,
    mesh_sequence_format: str,
    inference_python: Optional[str | Path] = None,
) -> Path:
    """Run canonical Stage 1 in the SAM3D inference environment."""
    print("\n" + "=" * 60)
    print("Stage 1: SAM3D Body Inference")
    print("=" * 60)

    if inference_python is None:
        inference_python = require_conda_env_python(
            "sam_3d_body",
            override_vars=("SAM3D_OPENSIM_SAM3D_PYTHON", "SAM3D_PYTHON"),
        )
    inference_python = Path(inference_python)

    cmd = [
        str(inference_python),
        "-m",
        "run_inference",
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--fps",
        str(fps),
        "--detector",
        detector,
        "--fov",
        fov,
    ]
    _append_bool_arg(cmd, "--single_person", single_person)
    _append_bool_arg(cmd, "--use-mask", use_mask)
    _append_bool_arg(cmd, "--save-mesh-video", save_mesh_video)
    _append_bool_arg(cmd, "--save-mesh-sequence", save_mesh_sequence)
    _append_optional_arg(cmd, "--config", config_path)
    _append_optional_arg(cmd, "--segmentor", segmentor or "none")
    _append_optional_arg(cmd, "--support-surface-mode", support_surface_mode)
    _append_optional_arg(cmd, "--mesh-sequence-format", mesh_sequence_format)

    print(f"  Detector: {detector}, FOV: {fov}, Segmentor: {segmentor or 'none'}")
    print(f"  Single-person selection: {'ENABLED' if single_person else 'disabled'}")
    print(f"  Support surface: {support_surface_mode}")
    print(f"  Save mesh video: {str(bool(save_mesh_video)).lower()}")
    print(
        "  Save mesh sequence: "
        f"{str(bool(save_mesh_sequence)).lower()} ({mesh_sequence_format})"
    )
    print(f"  Python: {inference_python}")

    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError("SAM3D Body inference failed")

    json_path = output_dir / "video_outputs.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Expected inference artifact missing: {json_path}")
    return json_path


def run_stage2_subprocess(
    *,
    project_root: Path,
    current_python: str | Path,
    json_path: Path,
    output_dir: Path,
    config_path: Optional[str] = None,
    subject_height: float,
    subject_mass: float,
    smooth: float,
    ground_alignment_mode: str,
    vertical_translation_mode: str,
    post_ik_foot_snap_mode: str,
    ik_backend: str = "direct_opensim",
    save_graph: bool,
    global_translation: bool,
    skip_ik: bool,
    skip_fbx: bool,
) -> dict[str, Optional[Path]]:
    """Run canonical Stage 2 in the current environment."""
    print("\n" + "=" * 60)
    print("Stage 2: Export / OpenSim IK / FBX")
    print("=" * 60)

    cmd = [
        str(current_python),
        "-m",
        "run_export",
        "--input",
        str(json_path),
        "--output",
        str(output_dir),
        "--height",
        str(subject_height),
        "--mass",
        str(subject_mass),
        "--smooth",
        str(smooth),
        "--ground-alignment-mode",
        ground_alignment_mode,
        "--vertical-translation-mode",
        vertical_translation_mode,
        "--post-ik-foot-snap",
        post_ik_foot_snap_mode,
        "--ik-backend",
        ik_backend,
    ]
    _append_optional_arg(cmd, "--config", config_path)
    _append_bool_arg(cmd, "--save_graph", save_graph)
    _append_bool_arg(cmd, "--global-translation", global_translation)
    _append_bool_arg(cmd, "--skip-ik", skip_ik)
    _append_bool_arg(cmd, "--skip-fbx", skip_fbx)

    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError("Export stage failed")

    return _discover_output_paths(output_dir)


def run_subprocess_pipeline(
    *,
    project_root: str | Path,
    workspace_root: Optional[str | Path] = None,
    current_python: str | Path,
    input_path: str | Path | None,
    config_path: Optional[str] = None,
    subject_height: float,
    subject_mass: float,
    fps: float,
    detector: str,
    segmentor: Optional[str],
    fov: str,
    use_mask: bool,
    single_person: bool,
    support_surface_mode: str,
    save_mesh_video: bool,
    save_mesh_sequence: bool,
    mesh_sequence_format: str,
    smooth: float,
    ground_alignment_mode: str,
    vertical_translation_mode: str,
    post_ik_foot_snap_mode: str,
    ik_backend: str = "direct_opensim",
    save_graph: bool,
    global_translation: bool,
    skip_inference: bool,
    skip_ik: bool,
    skip_fbx: bool,
    output_dir: Optional[str | Path] = None,
    inference_python: Optional[str | Path] = None,
) -> dict[str, Optional[Path]]:
    """Run the canonical two-stage pipeline across two Python environments."""
    project_root = Path(project_root)
    workspace_root = Path(workspace_root) if workspace_root is not None else project_root
    input_path = Path(input_path) if input_path is not None else None
    current_python = Path(current_python)

    if input_path is not None and not input_path.exists() and not skip_inference:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = resolve_pipeline_output_dir(
        input_path=input_path,
        output_dir=output_dir,
        base_output_dir=workspace_root,
    )
    print_pipeline_banner(
        title="SAM3D Body to OpenSim - Full Pipeline",
        input_path=input_path or "N/A (skip inference)",
        output_dir=output_dir,
        subject_height=subject_height,
        subject_mass=subject_mass,
        detector=detector,
        segmentor=segmentor,
        fov=fov,
        ground_alignment_mode=ground_alignment_mode,
        vertical_translation_mode=vertical_translation_mode,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        ik_backend=ik_backend,
        single_person=single_person,
        support_surface_mode=support_surface_mode,
        global_translation=global_translation,
    )

    if skip_inference:
        print("\nSkipping Stage 1 (using existing video_outputs.json)")
        json_path = output_dir / "video_outputs.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No existing video_outputs.json found in {output_dir}")
    else:
        if input_path is None:
            raise ValueError("input_path is required unless skip_inference is true")
        json_path = run_stage1_subprocess(
            project_root=project_root,
            input_path=input_path,
            output_dir=output_dir,
            config_path=config_path,
            fps=fps,
            detector=detector,
            segmentor=segmentor,
            fov=fov,
            use_mask=use_mask,
            single_person=single_person,
            support_surface_mode=support_surface_mode,
            save_mesh_video=save_mesh_video,
            save_mesh_sequence=save_mesh_sequence,
            mesh_sequence_format=mesh_sequence_format,
            inference_python=inference_python,
        )

    results = run_stage2_subprocess(
        project_root=project_root,
        current_python=current_python,
        json_path=json_path,
        output_dir=output_dir,
        config_path=config_path,
        subject_height=subject_height,
        subject_mass=subject_mass,
        smooth=smooth,
        ground_alignment_mode=ground_alignment_mode,
        vertical_translation_mode=vertical_translation_mode,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        ik_backend=ik_backend,
        save_graph=save_graph,
        global_translation=global_translation,
        skip_ik=skip_ik,
        skip_fbx=skip_fbx,
    )

    status_outputs = {
        name: (
            path
            if path and Path(path).exists()
            else f"MISSING: {path}"
        )
        for name, path in results.items()
    }
    print_pipeline_outputs_summary(status_outputs, output_dir=output_dir)

    return results
