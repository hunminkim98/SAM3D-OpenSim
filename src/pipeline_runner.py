"""Shared in-process composition of Stage 1 and Stage 2."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.export_stage import run_export_stage
from src.inference_stage import run_inference_stage
from src.pipeline_artifacts import build_processing_report, save_processing_report
from src.pipeline_runtime_common import (
    print_pipeline_banner,
    print_pipeline_outputs_summary,
    resolve_pipeline_output_dir,
    resolve_pipeline_runtime_options,
)


def run_combined_pipeline(
    *,
    input_path: str,
    project_root: str | Path,
    subject_height: Optional[float] = None,
    subject_mass: Optional[float] = None,
    output_dir: Optional[str] = None,
    target_fps: Optional[float] = None,
    device: Optional[str] = None,
    config_path: Optional[str] = None,
    skip_ik: bool = False,
    visualize: bool = False,
    save_fbx: bool = False,
    global_translation: bool = False,
    detector: Optional[str] = None,
    segmentor: Optional[str] = None,
    fov: Optional[str] = None,
    use_mask: bool = False,
    smooth_cutoff: Optional[float] = None,
    ground_alignment_mode: Optional[str] = None,
    vertical_translation_mode: Optional[str] = None,
    single_person: Optional[bool] = None,
    support_surface_mode: Optional[str] = None,
    post_ik_foot_snap_mode: Optional[str] = None,
    ik_backend: Optional[str] = None,
    save_graph: bool = False,
    save_mesh_video: bool = False,
    save_mesh_sequence: bool = False,
    mesh_sequence_format: str = "ply",
) -> dict:
    """Run Stage 1 then Stage 2 in-process using shared stage helpers."""
    start_time = time.time()
    results = {"success": False, "outputs": {}, "timings": {}}
    project_root = Path(project_root)

    resolved = resolve_pipeline_runtime_options(
        config_path=config_path,
        subject_height=subject_height,
        subject_mass=subject_mass,
        target_fps=target_fps,
        device=device,
        detector=detector,
        segmentor=segmentor,
        fov=fov,
        smooth_cutoff=smooth_cutoff,
        ground_alignment_mode=ground_alignment_mode,
        vertical_translation_mode=vertical_translation_mode,
        single_person=single_person,
        support_surface_mode=support_surface_mode,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        ik_backend=ik_backend,
    )
    output_dir = resolve_pipeline_output_dir(
        input_path=input_path,
        output_dir=output_dir,
    )
    results["outputs"]["directory"] = str(output_dir)
    print_pipeline_banner(
        title="SAM3D Body to OpenSim Pipeline",
        input_path=input_path,
        output_dir=output_dir,
        subject_height=resolved["subject_height"],
        subject_mass=resolved["subject_mass"],
        detector=resolved["detector"],
        segmentor=resolved["segmentor"],
        fov=resolved["fov"],
        ground_alignment_mode=resolved["ground_alignment_mode"],
        vertical_translation_mode=resolved["vertical_translation_mode"],
        post_ik_foot_snap_mode=resolved["post_ik_foot_snap_mode"],
        ik_backend=resolved["ik_backend"],
        single_person=resolved["single_person"],
        support_surface_mode=resolved["support_surface_mode"],
        global_translation=global_translation,
        device=resolved["device"],
    )

    if visualize:
        print("Visualization flag is deprecated and currently ignored.")

    t0 = time.time()
    stage1 = run_inference_stage(
        input_path=input_path,
        output_dir=str(output_dir),
        fps=resolved["target_fps"],
        device=resolved["device"],
        config_path=config_path,
        detector=resolved["detector"],
        segmentor=resolved["segmentor"],
        fov=resolved["fov"],
        use_mask=use_mask,
        single_person=resolved["single_person"],
        support_surface_mode=resolved["support_surface_mode"],
        vertical_translation_mode=resolved["vertical_translation_mode"],
        save_mesh_video=save_mesh_video,
        save_mesh_sequence=save_mesh_sequence,
        mesh_sequence_format=mesh_sequence_format,
        save_artifacts=True,
        save_raw_keypoints=True,
        show_header=False,
        step_offset=0,
        step_total=6,
    )
    results["timings"]["stage1_inference"] = time.time() - t0
    results["outputs"]["raw_keypoints"] = stage1["saved_paths"].get("raw_keypoints")
    results["outputs"]["video_outputs"] = stage1["saved_paths"]["video_outputs"]
    results["outputs"]["mesh_video"] = stage1["saved_paths"].get("mesh_video")
    results["outputs"]["mesh_sequence_dir"] = stage1["saved_paths"].get("mesh_sequence_dir")
    print(f"  Saved SAM3D format: {stage1['saved_paths']['video_outputs']}")
    if stage1["saved_paths"].get("mesh_video"):
        print(f"  Saved mesh overlay video: {stage1['saved_paths']['mesh_video']}")
    if stage1["saved_paths"].get("mesh_sequence_dir"):
        print(
            "  Saved mesh sequence: "
            f"{stage1['saved_paths']['mesh_sequence_dir']} "
            f"({stage1['saved_paths'].get('mesh_sequence_count', 0)} files)"
        )

    t0 = time.time()
    export_results = run_export_stage(
        json_path=stage1["saved_paths"]["video_outputs"],
        output_dir=str(output_dir),
        subject_height=resolved["subject_height"],
        subject_mass=resolved["subject_mass"],
        fps=stage1["actual_fps"],
        global_translation=global_translation,
        skip_ik=skip_ik,
        skip_fbx=not save_fbx,
        person_idx=0,
        smooth_cutoff=resolved["smooth_cutoff"],
        ground_alignment_mode=resolved["ground_alignment_mode"],
        vertical_translation_mode=resolved["vertical_translation_mode"],
        post_ik_foot_snap_mode=resolved["post_ik_foot_snap_mode"],
        ik_backend=resolved["ik_backend"],
        save_graph=save_graph,
        show_header=False,
        step_offset=2,
        step_total=6,
        project_root=project_root,
    )
    results["timings"]["stage2_export"] = time.time() - t0
    results["outputs"]["post_ik_contact_meta"] = export_results.get(
        "post_ik_contact_meta"
    )
    results["outputs"]["trc"] = export_results.get("trc")
    results["outputs"]["mot"] = export_results.get("mot")
    results["outputs"]["fbx"] = export_results.get("fbx")
    results["outputs"]["pose2sim_workspace"] = export_results.get("pose2sim_workspace")
    results["outputs"]["pose2sim_augmented_trc"] = export_results.get(
        "pose2sim_augmented_trc"
    )

    total_time = time.time() - start_time
    results["timings"]["total"] = total_time
    results["success"] = True

    report = build_processing_report(
        input_path=input_path,
        output_dir=output_dir,
        subject_height=resolved["subject_height"],
        subject_mass=resolved["subject_mass"],
        stage1=stage1,
        export_results=export_results,
        timings=results["timings"],
        visualize_requested=visualize,
        single_person=resolved["single_person"],
        post_ik_foot_snap_mode=resolved["post_ik_foot_snap_mode"],
        ik_backend=resolved["ik_backend"],
    )
    results["outputs"].update(report["outputs"])
    results["outputs"]["report"] = save_processing_report(report, output_dir)

    print_pipeline_outputs_summary(
        results["outputs"],
        total_time=total_time,
        frames_processed=len(stage1["frame_paths"]),
        valid_detections=int(stage1["valid_frames"].sum()),
    )

    return results
