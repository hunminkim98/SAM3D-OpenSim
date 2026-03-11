"""
Reusable Stage 2 orchestration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.blender_export import run_blender_fbx_export
from src.pipeline_artifacts import (
    extract_keypoints_and_cam,
    load_inference_meta,
    load_video_outputs,
)
from utils.io_utils import save_json


def _step_label(index: int, total: int, offset: int) -> str:
    return f"[{index + offset}/{total}]"


def run_export_stage(
    *,
    json_path: str,
    output_dir: str,
    subject_height: float,
    subject_mass: float,
    fps: float | None,
    global_translation: bool,
    skip_ik: bool,
    skip_fbx: bool,
    person_idx: int,
    smooth_cutoff: float = 6.0,
    ground_alignment_mode: str = "auto",
    vertical_translation_mode: str = "auto",
    post_ik_foot_snap_mode: str = "off",
    save_graph: bool = False,
    show_header: bool = True,
    header_title: str = "SAM3D Body Export to OpenSim",
    step_offset: int = 0,
    step_total: int = 5,
    project_root: str | Path | None = None,
) -> dict:
    """Load canonical stage-1 artifacts and export TRC/MOT/FBX outputs."""
    start_time = time.time()
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(project_root or Path(__file__).resolve().parent.parent)

    if show_header:
        print(f"\n{'=' * 60}")
        print(header_title)
        print(f"{'=' * 60}")
        print(f"Input: {json_path}")
        print(f"Output: {output_dir}")
        print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
        print(f"Global translation: {'ENABLED' if global_translation else 'disabled'}")
        print(f"Ground alignment: {ground_alignment_mode}")
        print(f"Vertical translation: {vertical_translation_mode}")
        print(f"Post-IK foot snap: {post_ik_foot_snap_mode}")
        print(f"{'=' * 60}\n")

    print(f"{_step_label(1, step_total, step_offset)} Loading SAM3D outputs...")
    data = load_video_outputs(json_path)
    meta = load_inference_meta(json_path)
    keypoints_3d, cam_translations, _focal_lengths, valid_frames = extract_keypoints_and_cam(
        data, person_idx
    )
    print(f"  Loaded {len(data)} frames, {int(np.sum(valid_frames))} valid detections")

    if fps is None:
        if meta:
            fps = meta.get("fps", 30.0)
            print(f"  FPS from metadata: {fps}")
        else:
            fps = 30.0
            print(f"  Using default FPS: {fps}")

    print(f"\n{_step_label(2, step_total, step_offset)} Post-processing keypoints...")
    from src.coordinate_transform import CoordinateTransformer
    from src.keypoint_converter import KeypointConverter
    from src.moge_scene_ground import extract_scene_ground_arrays_from_json
    from src.post_ik_foot_snap import build_post_ik_contact_meta
    from src.post_processing import PostProcessor
    from src.trc_exporter import TRCExporter

    use_smoothing = smooth_cutoff > 0
    if use_smoothing:
        print(f"  Smoothing enabled: {smooth_cutoff} Hz cutoff")

    post_processor = PostProcessor(
        smooth_filter=use_smoothing,
        filter_cutoff=smooth_cutoff,
        normalize_bones=True,
    )
    keypoints_processed = post_processor.process(keypoints_3d, fps=fps, subject_height=subject_height)

    scene_ground_data = extract_scene_ground_arrays_from_json(data, person_idx=person_idx)
    clip_scene_ground = meta.get("scene_ground", {}) if meta else {}
    if clip_scene_ground:
        scene_ground_data.update(
            {
                "normal_cam": clip_scene_ground.get("normal_cam"),
                "offset_cam": clip_scene_ground.get("offset_cam"),
                "clip_plane_confidence": clip_scene_ground.get("confidence"),
                "clip_plane_inlier_ratio": clip_scene_ground.get("inlier_ratio"),
                "support_surface_mode_applied": clip_scene_ground.get("support_surface_mode_applied"),
                "support_surface_selection_status": clip_scene_ground.get("support_surface_selection_status"),
            }
        )
    if scene_ground_data.get("available"):
        print(
            "  Loaded MoGe scene-ground hints "
            f"({int(np.sum(scene_ground_data['valid_frames']))} frames)"
        )

    transformer = CoordinateTransformer(subject_height=subject_height, units="mm")
    if global_translation:
        print("  Applying global translation from cam_t...")

    keypoints_opensim = transformer.transform(
        keypoints_processed,
        camera_translation=cam_translations if global_translation else None,
        center_pelvis=not global_translation,
        align_to_ground=True,
        apply_global_translation=global_translation,
        ground_alignment_mode=ground_alignment_mode,
        scene_ground_data=scene_ground_data,
        vertical_translation_mode=vertical_translation_mode,
    )
    ground_alignment_info = transformer.get_last_ground_alignment_info()
    ground_alignment_message = (
        "  Ground alignment applied: "
        f"{ground_alignment_info.get('applied_mode')} "
        f"(contact_frames={ground_alignment_info.get('contact_frames')}, "
        f"flight_frames={ground_alignment_info.get('flight_frames')})"
    )
    if ground_alignment_info.get("scene_ground_used"):
        ground_alignment_message += (
            f", scene_ground_fused_frames="
            f"{ground_alignment_info.get('scene_ground_fused_frames')}"
        )
    ground_alignment_message += (
        f", vertical_mode={ground_alignment_info.get('vertical_mode')}"
        f", vertical_confident_frames={ground_alignment_info.get('vertical_confident_frames')}"
    )
    if ground_alignment_info.get("manual_plane_anchor_active"):
        ground_alignment_message += (
            ", manual_anchor=on"
            f", manual_bias_l={ground_alignment_info.get('manual_plane_left_bias_m'):.3f}"
            f", manual_bias_r={ground_alignment_info.get('manual_plane_right_bias_m'):.3f}"
        )
    elif ground_alignment_info.get("manual_plane_fallback_reason"):
        ground_alignment_message += (
            ", manual_anchor=off"
            f", manual_reason={ground_alignment_info.get('manual_plane_fallback_reason')}"
        )
    print(ground_alignment_message)
    print("  Coordinate transformation complete")

    post_ik_contact_meta = build_post_ik_contact_meta(
        transformer.get_last_contact_data(),
        ground_alignment_info,
        fps=fps,
    )
    post_ik_contact_meta_path = output_dir / "post_ik_contact_meta.json"
    save_json(post_ik_contact_meta, post_ik_contact_meta_path)
    print(f"  Saved post-IK contact meta: {post_ik_contact_meta_path}")

    print(f"\n{_step_label(3, step_total, step_offset)} Converting to OpenSim markers...")
    converter = KeypointConverter(
        mapping_path=str(project_root / "config" / "marker_mapping.yaml")
    )
    markers, marker_names = converter.convert(keypoints_opensim, include_derived=True)
    print(f"  Generated {len(marker_names)} markers")

    print(f"\n{_step_label(4, step_total, step_offset)} Exporting TRC file...")
    video_name = json_path.parent.name
    if video_name in {".", ""}:
        video_name = json_path.stem.replace("video_outputs", "export")
    trc_exporter = TRCExporter(fps=fps, units="mm")
    trc_path = output_dir / f"markers_{video_name}.trc"
    trc_exporter.export(markers, marker_names, str(trc_path))
    print(f"  Saved: {trc_path}")

    results = {
        "trc": str(trc_path),
        "mot": None,
        "fbx": None,
        "ground_alignment": ground_alignment_info,
        "post_ik_contact_meta": str(post_ik_contact_meta_path),
        "graph_coords_dir": None,
        "graph_angles_dir": None,
    }

    from src.opensim_ik import run_external_opensim_ik

    if not skip_ik:
        print(f"\n{_step_label(5, step_total, step_offset)} Running OpenSim IK...")
        mot_path = run_external_opensim_ik(
            trc_path=trc_path,
            output_dir=output_dir,
            project_root=project_root,
            post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        )
        results["mot"] = str(mot_path) if mot_path else None
    else:
        print(f"\n{_step_label(5, step_total, step_offset)} Skipping OpenSim IK")

    if not skip_fbx and results["mot"]:
        print("\nExporting FBX...")
        fbx_path = run_blender_fbx_export(
            mot_path=results["mot"],
            output_dir=output_dir,
            project_root=project_root,
        )
        results["fbx"] = str(fbx_path) if fbx_path else None

    if save_graph:
        from src.export_graphs import save_export_graphs

        graph_results = save_export_graphs(
            trc_path=trc_path,
            mot_path=results["mot"],
            output_dir=output_dir,
        )
        results["graph_coords_dir"] = graph_results.get("coords_dir")
        results["graph_angles_dir"] = graph_results.get("angles_dir")

    elapsed = time.time() - start_time
    results["elapsed"] = elapsed
    return results
