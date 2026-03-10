"""
Scene-ground helpers powered by MoGe depth/point outputs.

This module keeps MoGe-specific geometry processing separate from the main
SAM3D wrapper and the OpenSim export path.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


FOOT_KEYPOINT_INDICES = {
    "left_toe": 15,
    "left_heel": 17,
    "right_toe": 18,
    "right_heel": 20,
}

LEFT_FOOT_KEYS = ("left_heel", "left_toe")
RIGHT_FOOT_KEYS = ("right_heel", "right_toe")

DEFAULT_CONTACT_CLEARANCE_M = 0.04
DEFAULT_FLIGHT_CLEARANCE_M = 0.08


def _clamp_roi_xyxy(
    roi_xyxy: np.ndarray,
    image_shape: tuple[int, ...],
) -> Optional[np.ndarray]:
    """Clamp an ROI to image bounds and reject degenerate rectangles."""
    height, width = image_shape[:2]
    roi = np.asarray(roi_xyxy, dtype=np.float32).reshape(-1)
    if roi.size != 4:
        return None

    x1 = float(np.clip(min(roi[0], roi[2]), 0, width - 1))
    y1 = float(np.clip(min(roi[1], roi[3]), 0, height - 1))
    x2 = float(np.clip(max(roi[0], roi[2]), 0, width - 1))
    y2 = float(np.clip(max(roi[1], roi[3]), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _expand_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, ...],
    scale: float = 1.15,
) -> np.ndarray:
    """Expand a bbox slightly to remove people from plane-fitting candidates."""
    height, width = image_shape[:2]
    bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
    center_x = (bbox[0] + bbox[2]) * 0.5
    center_y = (bbox[1] + bbox[3]) * 0.5
    half_w = max(1.0, (bbox[2] - bbox[0]) * 0.5 * scale)
    half_h = max(1.0, (bbox[3] - bbox[1]) * 0.5 * scale)
    expanded = np.array(
        [
            max(0.0, center_x - half_w),
            max(0.0, center_y - half_h),
            min(float(width - 1), center_x + half_w),
            min(float(height - 1), center_y + half_h),
        ],
        dtype=np.float32,
    )
    return expanded


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane with SVD and orient the normal toward camera-up (-Y)."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        raise ValueError("Degenerate plane fit.")
    normal /= norm
    if normal[1] > 0:
        normal *= -1.0
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def _fit_plane_ransac(
    points: np.ndarray,
    iterations: int = 96,
    distance_threshold: float = 0.04,
    min_vertical_component: float = 0.25,
) -> Optional[Dict[str, Any]]:
    """Robustly fit a floor-like plane from camera-space points."""
    if len(points) < 3:
        return None

    rng = np.random.default_rng(0)
    best_inliers = None
    best_score = -1.0

    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        vec_a = sample[1] - sample[0]
        vec_b = sample[2] - sample[0]
        normal = np.cross(vec_a, vec_b)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue

        normal = (normal / norm).astype(np.float32)
        if abs(float(normal[1])) < min_vertical_component:
            continue
        if normal[1] > 0:
            normal *= -1.0

        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(points @ normal + offset)
        inliers = distances <= distance_threshold
        inlier_ratio = float(np.mean(inliers))
        score = float(np.sum(inliers)) * (0.5 + 0.5 * abs(float(normal[1])))
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < 3:
        return None

    inlier_points = points[best_inliers]
    normal, offset = _fit_plane_svd(inlier_points)
    residuals = np.abs(inlier_points @ normal + offset)
    all_residuals = np.abs(points @ normal + offset)
    return {
        "normal_cam": normal,
        "offset_cam": offset,
        "num_inliers": int(np.sum(best_inliers)),
        "num_candidates": int(len(points)),
        "inlier_ratio": float(np.mean(best_inliers)),
        "residual_median": float(np.median(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "all_residual_median": float(np.median(all_residuals)),
    }


def estimate_ground_plane_from_scene_points(
    points: np.ndarray,
    mask: Optional[np.ndarray],
    image_shape: tuple[int, ...],
    exclude_bboxes: Optional[Iterable[np.ndarray]] = None,
    roi_xyxy: Optional[np.ndarray] = None,
    mode: str = "auto",
    lower_band_ratio: float = 0.35,
    max_candidate_points: int = 6000,
    min_candidate_points: int = 256,
    distance_threshold: float = 0.04,
    min_inlier_ratio: float = 0.12,
) -> Optional[Dict[str, Any]]:
    """
    Estimate a clip-level ground plane from MoGe camera-space points.

    The plane is fit either from a manual ROI or from the lower image band
    while excluding person bboxes.
    """
    scene_points = np.asarray(points, dtype=np.float32)
    if scene_points.ndim != 3 or scene_points.shape[-1] != 3:
        return None

    height, width = image_shape[:2]
    valid = np.isfinite(scene_points).all(axis=-1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    yy, xx = np.indices((height, width))
    source_mode = "manual-roi" if roi_xyxy is not None else "auto-lower-band"
    roi = None if roi_xyxy is None else _clamp_roi_xyxy(roi_xyxy, image_shape)
    if roi_xyxy is not None and roi is None:
        return None

    if roi is not None:
        valid &= (
            (xx >= int(np.floor(roi[0])))
            & (xx <= int(np.ceil(roi[2])))
            & (yy >= int(np.floor(roi[1])))
            & (yy <= int(np.ceil(roi[3])))
        )
    else:
        lower_start = int(height * (1.0 - lower_band_ratio))
        valid &= yy >= lower_start

    if exclude_bboxes:
        for bbox in exclude_bboxes:
            if bbox is None:
                continue
            x1, y1, x2, y2 = _expand_bbox(np.asarray(bbox, dtype=np.float32), image_shape)
            valid &= ~(
                (xx >= int(np.floor(x1)))
                & (xx <= int(np.ceil(x2)))
                & (yy >= int(np.floor(y1)))
                & (yy <= int(np.ceil(y2)))
            )

    candidate_mask = valid.copy()
    candidate_points = scene_points[valid]
    if len(candidate_points) < min_candidate_points:
        return None

    if len(candidate_points) > max_candidate_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(candidate_points), size=max_candidate_points, replace=False)
        candidate_points = candidate_points[keep]

    plane = _fit_plane_ransac(
        candidate_points,
        distance_threshold=distance_threshold,
    )
    if plane is None:
        return None
    if plane["inlier_ratio"] < min_inlier_ratio:
        return None

    signed_distance_map = np.abs(np.sum(scene_points * plane["normal_cam"], axis=-1) + plane["offset_cam"])
    inlier_mask = candidate_mask & (signed_distance_map <= distance_threshold)
    confidence = plane["inlier_ratio"] * max(0.0, min(1.0, abs(float(plane["normal_cam"][1]))))
    plane.update(
        {
            "confidence": float(confidence),
            "lower_band_ratio": float(lower_band_ratio),
            "distance_threshold_m": float(distance_threshold),
            "min_inlier_ratio": float(min_inlier_ratio),
            "source_mode": source_mode,
            "mode_requested": str(mode),
            "roi_xyxy": None if roi is None else roi.astype(float).tolist(),
            "candidate_mask_debug": candidate_mask,
            "inlier_mask_debug": inlier_mask,
        }
    )
    return plane


def compute_output_scene_ground(
    output: Dict[str, Any],
    plane_model: Optional[Dict[str, Any]],
    contact_clearance_m: float = DEFAULT_CONTACT_CLEARANCE_M,
    flight_clearance_m: float = DEFAULT_FLIGHT_CLEARANCE_M,
) -> Optional[Dict[str, Any]]:
    """Compute per-frame foot-to-ground clearances from a plane model."""
    if not plane_model:
        return None

    pred_keypoints_3d = np.asarray(output.get("pred_keypoints_3d"), dtype=np.float32)
    pred_cam_t = np.asarray(output.get("pred_cam_t"), dtype=np.float32).reshape(-1)
    if pred_keypoints_3d.ndim != 2 or pred_keypoints_3d.shape[1] != 3 or pred_cam_t.size != 3:
        return None

    normal = np.asarray(plane_model.get("normal_cam"), dtype=np.float32).reshape(-1)
    offset = float(plane_model.get("offset_cam", 0.0))
    if normal.size != 3:
        return None

    joints_cam = pred_keypoints_3d + pred_cam_t[None, :]
    clearances: Dict[str, float] = {}
    signed_distances: Dict[str, float] = {}

    for key, joint_idx in FOOT_KEYPOINT_INDICES.items():
        if joint_idx >= joints_cam.shape[0]:
            signed_distances[key] = float("nan")
            clearances[key] = float("nan")
            continue
        signed_distance = float(np.dot(joints_cam[joint_idx], normal) + offset)
        signed_distances[key] = signed_distance
        clearances[key] = max(0.0, signed_distance)

    left_candidates = [clearances[key] for key in LEFT_FOOT_KEYS if np.isfinite(clearances[key])]
    right_candidates = [clearances[key] for key in RIGHT_FOOT_KEYS if np.isfinite(clearances[key])]
    left_clearance = min(left_candidates) if left_candidates else float("nan")
    right_clearance = min(right_candidates) if right_candidates else float("nan")

    return {
        "left_heel_signed_distance": signed_distances["left_heel"],
        "left_toe_signed_distance": signed_distances["left_toe"],
        "right_heel_signed_distance": signed_distances["right_heel"],
        "right_toe_signed_distance": signed_distances["right_toe"],
        "left_heel_clearance": clearances["left_heel"],
        "left_toe_clearance": clearances["left_toe"],
        "right_heel_clearance": clearances["right_heel"],
        "right_toe_clearance": clearances["right_toe"],
        "left_clearance": left_clearance,
        "right_clearance": right_clearance,
        "left_contact_hint": bool(np.isfinite(left_clearance) and left_clearance <= contact_clearance_m),
        "right_contact_hint": bool(np.isfinite(right_clearance) and right_clearance <= contact_clearance_m),
        "left_flight_hint": bool(np.isfinite(left_clearance) and left_clearance >= flight_clearance_m),
        "right_flight_hint": bool(np.isfinite(right_clearance) and right_clearance >= flight_clearance_m),
        "plane_confidence": float(plane_model.get("confidence", 0.0)),
        "plane_frame_idx": int(plane_model.get("frame_idx", -1)),
        "plane_inlier_ratio": float(plane_model.get("inlier_ratio", 0.0)),
    }


def build_scene_ground_arrays(
    frame_source: List[Dict[str, Any]],
    output_getter,
) -> Dict[str, Any]:
    """Collect per-frame scene-ground metadata into array form."""
    num_frames = len(frame_source)
    left_clearance = np.full(num_frames, np.nan, dtype=np.float32)
    right_clearance = np.full(num_frames, np.nan, dtype=np.float32)
    plane_confidence = np.zeros(num_frames, dtype=np.float32)
    valid_frames = np.zeros(num_frames, dtype=bool)
    left_contact_hint = np.zeros(num_frames, dtype=bool)
    right_contact_hint = np.zeros(num_frames, dtype=bool)
    left_flight_hint = np.zeros(num_frames, dtype=bool)
    right_flight_hint = np.zeros(num_frames, dtype=bool)

    for idx, frame_data in enumerate(frame_source):
        scene_ground = output_getter(frame_data)
        if not scene_ground:
            continue

        left = scene_ground.get("left_clearance")
        right = scene_ground.get("right_clearance")
        if left is not None and np.isfinite(left):
            left_clearance[idx] = float(left)
        if right is not None and np.isfinite(right):
            right_clearance[idx] = float(right)
        plane_confidence[idx] = float(scene_ground.get("plane_confidence", 0.0))
        left_contact_hint[idx] = bool(scene_ground.get("left_contact_hint", False))
        right_contact_hint[idx] = bool(scene_ground.get("right_contact_hint", False))
        left_flight_hint[idx] = bool(scene_ground.get("left_flight_hint", False))
        right_flight_hint[idx] = bool(scene_ground.get("right_flight_hint", False))
        valid_frames[idx] = np.isfinite(left_clearance[idx]) or np.isfinite(right_clearance[idx])

    return {
        "available": bool(np.any(valid_frames)),
        "valid_frames": valid_frames,
        "left_clearance": left_clearance,
        "right_clearance": right_clearance,
        "left_contact_hint": left_contact_hint,
        "right_contact_hint": right_contact_hint,
        "left_flight_hint": left_flight_hint,
        "right_flight_hint": right_flight_hint,
        "plane_confidence": plane_confidence,
        "contact_clearance_m": DEFAULT_CONTACT_CLEARANCE_M,
        "flight_clearance_m": DEFAULT_FLIGHT_CLEARANCE_M,
    }


def extract_scene_ground_arrays_from_frame_outputs(
    frame_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract scene-ground arrays from in-memory inference frame outputs."""
    return build_scene_ground_arrays(
        frame_outputs,
        output_getter=lambda frame_data: (
            frame_data.get("output", {}) or {}
        ).get("scene_ground"),
    )


def extract_scene_ground_arrays_from_json(
    data: List[Dict[str, Any]],
    person_idx: int = 0,
) -> Dict[str, Any]:
    """Extract scene-ground arrays from saved video_outputs.json data."""
    def _output_getter(frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        outputs = frame_data.get("outputs", [])
        if len(outputs) <= person_idx:
            return None
        person = outputs[person_idx]
        return person.get("scene_ground")

    return build_scene_ground_arrays(data, output_getter=_output_getter)
