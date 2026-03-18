"""
Reusable Stage 1 orchestration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from utils.io_utils import load_config
from utils.video_utils import extract_frames, get_video_info

from src.pipeline_artifacts import (
    build_inference_meta,
    build_video_outputs_from_inference,
    save_inference_artifacts,
)
from src.mesh_visualization import save_mesh_sidecars


def _step_label(index: int, total: int, offset: int) -> str:
    return f"[{index + offset}/{total}]"


def _resolve_component_name(value: Optional[str]) -> Optional[str]:
    return value if value and value != "none" else None


def run_inference_stage(
    *,
    input_path: str,
    output_dir: str,
    fps: float,
    device: str,
    config_path: Optional[str] = None,
    detector: str = "vitdet",
    segmentor: Optional[str] = None,
    fov: str = "moge2",
    use_mask: bool = False,
    single_person: bool = True,
    support_surface_mode: Optional[str] = None,
    vertical_translation_mode: Optional[str] = None,
    save_mesh_video: bool = False,
    save_mesh_sequence: bool = False,
    mesh_sequence_format: str = "ply",
    save_artifacts: bool = True,
    save_raw_keypoints: bool = False,
    show_header: bool = True,
    header_title: str = "SAM3D Body Inference",
    step_offset: int = 0,
    step_total: int = 2,
) -> dict:
    """Run frame extraction and SAM3D inference, optionally saving artifacts."""
    start_time = time.time()
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_support_surface_mode = (
        support_surface_mode
        or config.get("processing", {}).get("support_surface_mode", "auto")
    )
    detector_name = _resolve_component_name(detector)
    segmentor_name = _resolve_component_name(segmentor)
    fov_name = _resolve_component_name(fov)

    if show_header:
        print(f"\n{'=' * 60}")
        print(header_title)
        print(f"{'=' * 60}")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Device: {device}")
        print(f"Detector: {detector_name or 'none'}")
        print(f"Segmentor: {segmentor_name or 'none'}")
        print(f"FOV estimator: {fov_name or 'none'}")
        print(f"Use mask: {use_mask}")
        print(f"Single-person selection: {'ENABLED' if single_person else 'disabled'}")
        print(f"Support surface: {resolved_support_surface_mode}")
        print(f"{'=' * 60}\n")

    print(f"{_step_label(1, step_total, step_offset)} Extracting frames...")
    video_info = get_video_info(input_path)
    print(f"  Video: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS")
    frames_dir = output_dir / "frames"
    frame_paths, actual_fps = extract_frames(input_path, str(frames_dir), target_fps=fps)
    print(f"  Extracted {len(frame_paths)} frames at {actual_fps:.2f} FPS")

    print(f"\n{_step_label(2, step_total, step_offset)} Running SAM3D Body inference...")
    from src.sam3d_inference import SAM3DInference

    sam3d = SAM3DInference(
        sam3d_root=config["sam3d"]["sam3d_root"],
        checkpoint_path=config["sam3d"]["checkpoint"],
        mhr_path=config["sam3d"]["mhr_path"],
        device=device,
        detector_name=detector_name,
        detector_path=config["sam3d"].get("detector_path"),
        segmentor_name=segmentor_name,
        segmentor_path=config["sam3d"].get("segmentor_path"),
        fov_name=fov_name,
        fov_path=config["sam3d"].get("fov_path"),
        bbox_threshold=config["sam3d"]["bbox_threshold"],
        use_mask=use_mask,
        inference_type=config["sam3d"]["inference_type"],
        single_person=single_person,
        support_surface_mode=resolved_support_surface_mode,
    )
    inference_results = sam3d.process_video(frame_paths, progress=True)
    keypoints_3d, valid_frames = sam3d.extract_keypoints_3d(inference_results["frames"])
    camera_params = sam3d.extract_camera_params(inference_results["frames"])
    print(
        f"  Processed {inference_results['num_frames']} frames, "
        f"{int(valid_frames.sum())} valid detections"
    )

    video_outputs = build_video_outputs_from_inference(
        inference_results,
        video_info,
        keypoints_3d=keypoints_3d,
        camera_params=camera_params,
        single_person=single_person,
    )
    inference_meta = build_inference_meta(
        input_video=str(input_path),
        fps=actual_fps,
        num_frames=len(frame_paths),
        video_info=video_info,
        inference_time=time.time() - start_time,
        single_person=single_person,
        support_surface_mode=resolved_support_surface_mode,
        vertical_translation_mode=vertical_translation_mode,
        selection=inference_results.get("selection", {}),
        scene_ground=inference_results.get("scene_ground", {}),
    )

    raw_payload = None
    if save_raw_keypoints:
        raw_payload = {
            "keypoints_3d": keypoints_3d.tolist(),
            "valid_frames": valid_frames.tolist(),
            "camera_params": {key: value.tolist() for key, value in camera_params.items()},
            "fps": actual_fps,
        }

    saved_paths = {}
    if save_artifacts:
        saved_paths = save_inference_artifacts(
            output_dir=output_dir,
            video_outputs=video_outputs,
            inference_meta=inference_meta,
            raw_payload=raw_payload,
        )

    if save_mesh_video or save_mesh_sequence:
        print("\nGenerating Stage 1 mesh sidecar outputs...")
        print("  Running mesh-quality full-refresh pass (no tracking fast path, no focal cache)...")
        mesh_frame_outputs = sam3d.process_video_for_mesh_sidecars(
            frame_paths,
            progress=True,
        )
        mesh_saved_paths = save_mesh_sidecars(
            output_dir=output_dir,
            frame_paths=frame_paths,
            frame_outputs=mesh_frame_outputs,
            faces=sam3d.faces,
            fps=actual_fps,
            single_person=single_person,
            save_mesh_video=save_mesh_video,
            save_mesh_sequence_files=save_mesh_sequence,
            mesh_sequence_format=mesh_sequence_format,
        )
        saved_paths.update(mesh_saved_paths)
        if mesh_saved_paths.get("mesh_video"):
            print(f"  Saved mesh overlay video: {mesh_saved_paths['mesh_video']}")
        elif mesh_saved_paths.get("mesh_video_error"):
            print(
                "  Warning: Mesh overlay video was requested but could not be generated: "
                f"{mesh_saved_paths['mesh_video_error']}"
            )
        if mesh_saved_paths.get("mesh_sequence_dir"):
            print(
                "  Saved mesh sequence: "
                f"{mesh_saved_paths['mesh_sequence_dir']} "
                f"({mesh_saved_paths.get('mesh_sequence_count', 0)} files, "
                f"format={mesh_saved_paths.get('mesh_sequence_format')})"
            )
        elif mesh_saved_paths.get("mesh_sequence_error"):
            print(
                "  Warning: Mesh sequence export was requested but could not be generated: "
                f"{mesh_saved_paths['mesh_sequence_error']}"
            )

    return {
        "output_dir": str(output_dir),
        "config": config,
        "video_info": video_info,
        "actual_fps": actual_fps,
        "frame_paths": frame_paths,
        "inference_results": inference_results,
        "keypoints_3d": keypoints_3d,
        "valid_frames": valid_frames,
        "camera_params": camera_params,
        "video_outputs": video_outputs,
        "inference_meta": inference_meta,
        "saved_paths": saved_paths,
        "support_surface_mode": resolved_support_surface_mode,
        "vertical_translation_mode": vertical_translation_mode,
    }
