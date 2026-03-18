"""
Shared helpers for pipeline artifacts and schema handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.io_utils import load_json, save_json


def load_video_outputs(json_path: str | Path) -> list:
    """Load SAM3D per-frame outputs."""
    data = load_json(str(json_path))
    if not isinstance(data, list):
        raise ValueError("Expected list of frame outputs")
    return data


def load_inference_meta(json_path: str | Path) -> dict:
    """Load optional inference metadata adjacent to video_outputs.json."""
    json_path = Path(json_path)
    meta_path = json_path.parent / "inference_meta.json"
    if not meta_path.exists():
        return {}
    meta = load_json(str(meta_path))
    return meta if isinstance(meta, dict) else {}


def extract_keypoints_and_cam(data: list, person_idx: int = 0) -> tuple:
    """
    Extract keypoints and camera params from SAM3D JSON data.

    Returns:
        keypoints_3d: (N, 70, 3) array
        cam_translations: (N, 3) array
        focal_lengths: (N,) array
        valid_frames: (N,) bool array
    """
    num_frames = len(data)
    keypoints_3d = np.zeros((num_frames, 70, 3), dtype=np.float32)
    cam_translations = np.zeros((num_frames, 3), dtype=np.float32)
    focal_lengths = np.zeros(num_frames, dtype=np.float32)
    valid_frames = np.zeros(num_frames, dtype=bool)

    for i, frame_data in enumerate(data):
        outputs = frame_data.get("outputs", [])
        if len(outputs) <= person_idx:
            continue

        person = outputs[person_idx]
        kp3d = person.get("pred_keypoints_3d", [])
        if len(kp3d) == 70:
            keypoints_3d[i] = np.array(kp3d, dtype=np.float32)
            valid_frames[i] = True

        cam_t = person.get("pred_cam_t", [0, 0, 5])
        cam_translations[i] = np.array(cam_t, dtype=np.float32)
        focal_lengths[i] = float(person.get("focal_length", 1000.0))

    return keypoints_3d, cam_translations, focal_lengths, valid_frames


def build_video_outputs_from_inference(
    inference_results: dict,
    video_info: dict,
    *,
    keypoints_3d: Optional[np.ndarray] = None,
    camera_params: Optional[dict] = None,
    single_person: bool = True,
) -> list:
    """Convert in-memory inference outputs to the canonical JSON schema."""
    frames = inference_results.get("frames", [])
    outputs = []

    for frame_index, frame_data in enumerate(frames):
        frame_name = Path(frame_data["frame_path"]).name
        frame_entry = {"frame": frame_name, "outputs": []}

        if single_person:
            selected = frame_data.get("output")
            outputs_to_save = [selected] if selected is not None else []
        else:
            outputs_to_save = frame_data.get("outputs", [])

        for output_idx, out in enumerate(outputs_to_save):
            if single_person and output_idx == 0 and keypoints_3d is not None and camera_params is not None:
                pred_keypoints_3d = keypoints_3d[frame_index].tolist()
                pred_cam_t = camera_params["cam_translations"][frame_index].tolist()
            else:
                pred_keypoints_3d = np.array(out.get("pred_keypoints_3d", [])).tolist()
                pred_cam_t = np.array(out.get("pred_cam_t", [0, 0, 5])).tolist()

            person_data = {
                "bbox": out.get(
                    "bbox",
                    [0, 0, video_info["width"], video_info["height"]],
                ),
                "focal_length": float(out.get("focal_length", 1000.0)),
                "pred_keypoints_3d": pred_keypoints_3d,
                "pred_cam_t": pred_cam_t,
            }
            if "pred_keypoints_2d" in out:
                person_data["pred_keypoints_2d"] = np.array(out["pred_keypoints_2d"]).tolist()
            if "shape_params" in out:
                person_data["shape_params"] = np.array(out["shape_params"]).tolist()
            if "scene_ground" in out:
                person_data["scene_ground"] = out["scene_ground"]

            frame_entry["outputs"].append(person_data)

        outputs.append(frame_entry)

    return outputs


def build_inference_meta(
    *,
    input_video: str,
    fps: float,
    num_frames: int,
    video_info: dict,
    inference_time: Optional[float] = None,
    single_person: bool = True,
    support_surface_mode: Optional[str] = None,
    vertical_translation_mode: Optional[str] = None,
    selection: Optional[dict] = None,
    scene_ground: Optional[dict] = None,
    ground_alignment: Optional[dict] = None,
) -> dict:
    """Build the canonical inference metadata payload."""
    meta = {
        "input_video": str(input_video),
        "fps": fps,
        "num_frames": num_frames,
        "video_info": video_info,
        "single_person": single_person,
        "selection": selection or {},
        "scene_ground": scene_ground or {},
    }
    if inference_time is not None:
        meta["inference_time"] = inference_time
    if support_surface_mode is not None:
        meta["support_surface_mode"] = support_surface_mode
    if vertical_translation_mode is not None:
        meta["vertical_translation_mode"] = vertical_translation_mode
    if ground_alignment is not None:
        meta["ground_alignment"] = ground_alignment
    return meta


def save_inference_artifacts(
    *,
    output_dir: str | Path,
    video_outputs: list,
    inference_meta: dict,
    raw_payload: Optional[Dict[str, Any]] = None,
) -> dict:
    """Persist standard stage-1 artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_outputs_path = output_dir / "video_outputs.json"
    save_json(video_outputs, video_outputs_path)

    inference_meta_path = output_dir / "inference_meta.json"
    save_json(inference_meta, inference_meta_path)

    saved = {
        "video_outputs": str(video_outputs_path),
        "inference_meta": str(inference_meta_path),
    }

    if raw_payload is not None:
        raw_path = output_dir / "keypoints_raw.json"
        save_json(raw_payload, raw_path)
        saved["raw_keypoints"] = str(raw_path)

    return saved


def build_processing_report(
    *,
    input_path: str,
    output_dir: str | Path,
    subject_height: float,
    subject_mass: float,
    stage1: dict,
    export_results: dict,
    timings: dict,
    visualize_requested: bool = False,
    single_person: bool = True,
    post_ik_foot_snap_mode: str = "off",
    ik_backend: str = "direct_opensim",
) -> dict:
    """Build the combined processing report emitted by run_pipeline.py."""
    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "subject": {
            "height": float(subject_height),
            "mass": float(subject_mass),
        },
        "video_info": stage1["video_info"],
        "processing": {
            "fps": stage1["actual_fps"],
            "num_frames": len(stage1["frame_paths"]),
            "valid_frames": int(stage1["valid_frames"].sum()),
            "single_person": bool(single_person),
            "visualize_requested": bool(visualize_requested),
            "ik_backend": ik_backend,
            "ground_alignment": export_results.get("ground_alignment", {}),
            "post_ik_foot_snap_mode": post_ik_foot_snap_mode,
        },
        "timings": timings,
        "outputs": {
            "directory": str(output_dir),
            "raw_keypoints": stage1["saved_paths"].get("raw_keypoints"),
            "video_outputs": stage1["saved_paths"]["video_outputs"],
            "mesh_video": stage1["saved_paths"].get("mesh_video"),
            "mesh_sequence_dir": stage1["saved_paths"].get("mesh_sequence_dir"),
            "mesh_sequence_format": stage1["saved_paths"].get("mesh_sequence_format"),
            "mesh_sequence_count": stage1["saved_paths"].get("mesh_sequence_count"),
            "post_ik_contact_meta": export_results.get("post_ik_contact_meta"),
            "trc": export_results.get("trc"),
            "mot": export_results.get("mot"),
            "fbx": export_results.get("fbx"),
            "pose2sim_workspace": export_results.get("pose2sim_workspace"),
            "pose2sim_augmented_trc": export_results.get("pose2sim_augmented_trc"),
        },
        "selection": stage1["inference_results"].get("selection", {}),
    }


def save_processing_report(report: dict, output_dir: str | Path) -> str:
    """Persist the combined processing report in the standard location."""
    output_dir = Path(output_dir)
    report_path = output_dir / "processing_report.json"
    save_json(report, report_path)
    return str(report_path)
