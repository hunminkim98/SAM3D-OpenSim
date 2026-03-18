"""Shared orchestration helpers for in-process and subprocess pipeline runners."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.io_utils import get_output_dir
from utils.pipeline_options import load_cli_defaults


def resolve_pipeline_runtime_options(
    *,
    config_path: Optional[str] = None,
    subject_height: Optional[float] = None,
    subject_mass: Optional[float] = None,
    target_fps: Optional[float] = None,
    device: Optional[str] = None,
    detector: Optional[str] = None,
    segmentor: Optional[str] = None,
    fov: Optional[str] = None,
    smooth_cutoff: Optional[float] = None,
    ground_alignment_mode: Optional[str] = None,
    vertical_translation_mode: Optional[str] = None,
    single_person: Optional[bool] = None,
    support_surface_mode: Optional[str] = None,
    post_ik_foot_snap_mode: Optional[str] = None,
    ik_backend: Optional[str] = None,
) -> dict:
    """Resolve runtime options against config defaults."""
    defaults = load_cli_defaults(config_path)
    return {
        "subject_height": (
            float(subject_height) if subject_height is not None else defaults["height"]
        ),
        "subject_mass": (
            float(subject_mass) if subject_mass is not None else defaults["mass"]
        ),
        "target_fps": (
            float(target_fps) if target_fps is not None else defaults["fps"]
        ),
        "device": device or defaults["device"],
        "detector": detector if detector is not None else defaults["detector"],
        "segmentor": segmentor if segmentor is not None else defaults["segmentor"],
        "fov": fov if fov is not None else defaults["fov"],
        "single_person": (
            single_person if single_person is not None else defaults["single_person"]
        ),
        "ground_alignment_mode": (
            ground_alignment_mode or defaults["ground_alignment_mode"]
        ),
        "vertical_translation_mode": (
            vertical_translation_mode or defaults["vertical_translation_mode"]
        ),
        "support_surface_mode": (
            support_surface_mode or defaults["support_surface_mode"]
        ),
        "post_ik_foot_snap_mode": (
            post_ik_foot_snap_mode or defaults["post_ik_foot_snap_mode"]
        ),
        "ik_backend": (
            ik_backend or defaults["ik_backend"]
        ),
        "smooth_cutoff": (
            float(smooth_cutoff) if smooth_cutoff is not None else defaults["smooth"]
        ),
    }


def resolve_pipeline_output_dir(
    *,
    input_path: str | Path | None = None,
    output_dir: Optional[str | Path] = None,
    base_output_dir: Optional[str | Path] = None,
) -> Path:
    """Resolve the output directory for a pipeline run."""
    if output_dir is not None:
        resolved = Path(output_dir)
    elif input_path is not None and base_output_dir is not None:
        resolved = Path(base_output_dir) / get_output_dir(str(input_path))
    elif input_path is not None:
        resolved = get_output_dir(str(input_path))
    else:
        raise ValueError("output_dir is required when input_path is not provided")

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def print_pipeline_banner(
    *,
    title: str,
    input_path: str | Path,
    output_dir: str | Path,
    subject_height: float,
    subject_mass: float,
    detector: str,
    segmentor: Optional[str],
    fov: str,
    ground_alignment_mode: str,
    vertical_translation_mode: str,
    post_ik_foot_snap_mode: str,
    ik_backend: str,
    single_person: bool,
    support_surface_mode: str,
    global_translation: bool,
    device: Optional[str] = None,
) -> None:
    """Print a standard pipeline configuration banner."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
    if device is not None:
        print(f"Device: {device}")
    print(
        f"Detector: {detector if detector != 'none' else 'none'}, "
        f"FOV: {fov if fov != 'none' else 'none'}, "
        f"Segmentor: {segmentor or 'none'}"
    )
    print(f"Ground alignment: {ground_alignment_mode}")
    print(f"IK backend: {ik_backend}")
    print(f"Post-IK foot snap: {post_ik_foot_snap_mode}")
    print(f"Vertical translation: {vertical_translation_mode}")
    print(f"Single-person selection: {'ENABLED' if single_person else 'disabled'}")
    print(f"Support surface: {support_surface_mode}")
    print(f"Global translation: {'ENABLED' if global_translation else 'disabled'}")
    print(f"{'='*60}\n")


def print_pipeline_outputs_summary(
    outputs: dict,
    *,
    total_time: Optional[float] = None,
    frames_processed: Optional[int] = None,
    valid_detections: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Print a standard output summary for pipeline runners."""
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    if total_time is not None:
        print(f"Total time: {total_time:.2f}s")
    if frames_processed is not None:
        print(f"Frames processed: {frames_processed}")
    if valid_detections is not None:
        print(f"Valid detections: {valid_detections}")
    print(f"\nOutputs:")
    for key, path in outputs.items():
        if path:
            print(f"  {key}: {path}")
    if output_dir is not None:
        print(f"\nAll outputs in: {output_dir}")
    print(f"{'='*60}\n")
