"""
Post-IK MOT correction for stance-phase foot-ground snapping.

This module keeps OpenSim IK marker fitting untouched and only adjusts
`pelvis_ty` after IK so stance frames land closer to the ground plane.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


LEFT_FOOT_MARKERS: Tuple[str, ...] = ("LHeel", "LBigToe", "LSmallToe")
RIGHT_FOOT_MARKERS: Tuple[str, ...] = ("RHeel", "RBigToe", "RSmallToe")
FOOT_MARKERS: Tuple[str, ...] = (*LEFT_FOOT_MARKERS, *RIGHT_FOOT_MARKERS)
FOOT_MARKER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "LHeel": ("LHeel", "L_calc_study"),
    "LBigToe": ("LBigToe", "L_toe_study"),
    "LSmallToe": ("LSmallToe", "L_5meta_study"),
    "RHeel": ("RHeel", "r_calc_study"),
    "RBigToe": ("RBigToe", "r_toe_study"),
    "RSmallToe": ("RSmallToe", "r_5meta_study"),
}


def build_post_ik_contact_meta(
    contact_data: Optional[Dict[str, Any]],
    ground_alignment_info: Optional[Dict[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    """Build a serializable contact payload for post-IK MOT correction."""
    payload: Dict[str, Any] = {
        "available": False,
        "fps": float(fps),
        "frame_count": 0,
        "left_contact": [],
        "right_contact": [],
        "any_contact": [],
        "flight": [],
        "ground_alignment_applied_mode": None,
        "scene_ground_used": False,
        "manual_plane_anchor_active": False,
        "manual_plane_calibration_confidence": 0.0,
    }
    if not contact_data:
        return payload

    left_contact = np.asarray(contact_data.get("left_contact", []), dtype=bool)
    right_contact = np.asarray(contact_data.get("right_contact", []), dtype=bool)
    any_contact = np.asarray(contact_data.get("any_contact", []), dtype=bool)
    flight = np.asarray(contact_data.get("flight", []), dtype=bool)
    if (
        left_contact.size == 0
        or right_contact.shape != left_contact.shape
        or any_contact.shape != left_contact.shape
        or flight.shape != left_contact.shape
    ):
        return payload

    info = ground_alignment_info or {}
    payload.update(
        {
            "available": True,
            "frame_count": int(left_contact.shape[0]),
            "left_contact": left_contact.astype(bool).tolist(),
            "right_contact": right_contact.astype(bool).tolist(),
            "any_contact": any_contact.astype(bool).tolist(),
            "flight": flight.astype(bool).tolist(),
            "ground_alignment_applied_mode": info.get("applied_mode"),
            "scene_ground_used": bool(info.get("scene_ground_used", False)),
            "manual_plane_anchor_active": bool(
                info.get("manual_plane_anchor_active", False)
            ),
            "manual_plane_calibration_confidence": float(
                info.get("manual_plane_calibration_confidence", 0.0)
            ),
        }
    )
    return payload


def _read_contact_meta(contact_meta_path: Path) -> Dict[str, Any]:
    if not contact_meta_path.exists():
        return {"available": False, "reason": "missing-contact-meta"}
    with open(contact_meta_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {"available": False, "reason": "invalid-contact-meta"}
    return payload


def _read_mot(path: Path) -> Tuple[List[str], List[str], np.ndarray, bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    endheader_idx = None
    in_degrees = True
    for idx, line in enumerate(lines):
        lowered = line.strip().lower()
        if "indegrees=yes" in lowered:
            in_degrees = True
        elif "indegrees=no" in lowered:
            in_degrees = False
        if line.strip().lower() == "endheader":
            endheader_idx = idx
            break
    if endheader_idx is None or endheader_idx + 2 > len(lines):
        raise ValueError(f"Could not parse MOT header: {path}")

    header_lines = lines[: endheader_idx + 1]
    labels = lines[endheader_idx + 1].split()
    data_rows: List[List[float]] = []
    for raw_line in lines[endheader_idx + 2 :]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != len(labels):
            raise ValueError(
                f"Unexpected MOT row width in {path}: expected {len(labels)}, got {len(parts)}"
            )
        data_rows.append([float(value) for value in parts])

    data = np.asarray(data_rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(labels):
        raise ValueError(f"Invalid MOT data matrix in {path}")
    return header_lines, labels, data, in_degrees


def _write_mot(path: Path, header_lines: Sequence[str], labels: Sequence[str], data: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in header_lines:
            handle.write(line.rstrip("\n") + "\n")
        handle.write("\t".join(labels) + "\n")
        for row in data:
            handle.write("\t".join(f"{float(value):.8f}" for value in row) + "\n")


def _contiguous_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    signal = np.asarray(mask, dtype=bool)
    if signal.size == 0 or not np.any(signal):
        return []
    indices = np.flatnonzero(signal)
    start = int(indices[0])
    prev = int(indices[0])
    segments: List[Tuple[int, int]] = []
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        segments.append((start, prev))
        start = idx
        prev = idx
    segments.append((start, prev))
    return segments


def _smooth_contact_drop(raw_drop: np.ndarray, contact_mask: np.ndarray, transition_window: int) -> np.ndarray:
    drop = np.asarray(raw_drop, dtype=np.float32).copy()
    contact_mask = np.asarray(contact_mask, dtype=bool)
    smoothed = np.zeros_like(drop)
    for start, end in _contiguous_true_segments(contact_mask):
        seg = drop[start : end + 1]
        if seg.size >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            seg = np.convolve(seg, kernel, mode="same")
        smoothed[start : end + 1] = seg

        if transition_window > 0:
            for idx in range(start, min(end + 1, start + transition_window)):
                alpha = float(idx - start + 1) / float(transition_window + 1)
                smoothed[idx] *= alpha
            for idx in range(max(start, end - transition_window + 1), end + 1):
                alpha = float(end - idx + 1) / float(transition_window + 1)
                smoothed[idx] *= alpha

    return smoothed


def _compute_marker_heights_from_mot(
    model_path: Path,
    mot_path: Path,
    marker_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    import opensim as osim

    header_lines, labels, data, in_degrees = _read_mot(mot_path)
    del header_lines

    if not labels or labels[0].lower() != "time":
        raise ValueError(f"Expected MOT first column to be time: {mot_path}")

    model = osim.Model(str(model_path))
    state = model.initSystem()
    marker_set = model.getMarkerSet()
    coord_set = model.getCoordinateSet()

    coord_indices: Dict[int, Any] = {}
    for idx, label in enumerate(labels[1:], start=1):
        try:
            coord_indices[idx] = coord_set.get(label)
        except Exception:
            continue

    resolved_marker_names = _resolve_marker_name_aliases(marker_set, marker_names)
    marker_objs: Dict[str, Any] = {}
    for logical_name, actual_name in resolved_marker_names.items():
        try:
            marker_objs[logical_name] = marker_set.get(actual_name)
        except Exception as exc:
            raise ValueError(
                f"Marker {logical_name} (resolved to {actual_name}) not found in model "
                f"{model_path}: {exc}"
            ) from exc

    heights = {
        name: np.full(data.shape[0], np.nan, dtype=np.float32)
        for name in marker_names
    }

    for row_idx in range(data.shape[0]):
        row = data[row_idx]
        state.setTime(float(row[0]))
        for col_idx, coord in coord_indices.items():
            value = float(row[col_idx])
            if in_degrees and labels[col_idx] not in {"pelvis_tx", "pelvis_ty", "pelvis_tz"}:
                value = float(np.deg2rad(value))
            coord.setValue(state, value, False)
        model.realizePosition(state)
        for name, marker in marker_objs.items():
            location = marker.getLocationInGround(state)
            heights[name][row_idx] = float(location.get(1))

    return heights


def _resolve_marker_name_aliases(marker_set: Any, marker_names: Sequence[str]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for logical_name in marker_names:
        alias_candidates = FOOT_MARKER_ALIASES.get(logical_name, (logical_name,))
        for candidate_name in alias_candidates:
            try:
                marker_set.get(candidate_name)
                resolved[logical_name] = candidate_name
                break
            except Exception:
                continue
        if logical_name not in resolved:
            alias_text = ", ".join(alias_candidates)
            raise ValueError(
                f"Marker {logical_name} not found in model marker set. "
                f"Tried aliases: {alias_text}"
            )
    return resolved


def _compute_stance_drop_series(
    marker_heights: Dict[str, np.ndarray],
    contact_meta: Dict[str, Any],
    target_clearance_m: float,
    max_drop_m: float,
    transition_window: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    left_contact = np.asarray(contact_meta.get("left_contact", []), dtype=bool)
    right_contact = np.asarray(contact_meta.get("right_contact", []), dtype=bool)
    any_contact = np.asarray(contact_meta.get("any_contact", []), dtype=bool)
    if (
        left_contact.size == 0
        or right_contact.shape != left_contact.shape
        or any_contact.shape != left_contact.shape
    ):
        return np.zeros(0, dtype=np.float32), {"reason": "invalid-contact-arrays"}

    num_frames = left_contact.shape[0]
    raw_drop = np.zeros(num_frames, dtype=np.float32)
    support_height = np.full(num_frames, np.nan, dtype=np.float32)
    support_marker_name: List[Optional[str]] = [None] * num_frames

    for idx in range(num_frames):
        if not any_contact[idx]:
            continue

        candidate_markers: List[str] = []
        if left_contact[idx]:
            candidate_markers.extend(LEFT_FOOT_MARKERS)
        if right_contact[idx]:
            candidate_markers.extend(RIGHT_FOOT_MARKERS)
        if not candidate_markers:
            continue

        valid_values: List[Tuple[str, float]] = []
        for marker_name in candidate_markers:
            marker_series = marker_heights.get(marker_name)
            if marker_series is None or idx >= marker_series.shape[0]:
                continue
            marker_y = float(marker_series[idx])
            if np.isfinite(marker_y):
                valid_values.append((marker_name, marker_y))
        if not valid_values:
            continue

        marker_name, min_height = min(valid_values, key=lambda item: item[1])
        support_marker_name[idx] = marker_name
        support_height[idx] = float(min_height)
        raw_drop[idx] = float(
            np.clip(min_height - target_clearance_m, 0.0, max_drop_m)
        )

    smoothed_drop = _smooth_contact_drop(raw_drop, any_contact, transition_window)
    diagnostics = {
        "support_height": support_height,
        "support_marker_name": support_marker_name,
        "raw_drop": raw_drop,
    }
    return smoothed_drop, diagnostics


def apply_post_ik_foot_snap(
    model_path: str | Path,
    mot_path: str | Path,
    output_dir: str | Path,
    contact_meta_path: str | Path,
    mode: str = "off",
    target_clearance_m: float = 0.01,
    max_drop_m: float = 0.04,
    transition_window: int = 3,
) -> Dict[str, Any]:
    """
    Lower pelvis_ty during stance frames to reduce visible foot hover.

    The raw IK MOT is preserved as `*_raw.mot`; the corrected file overwrites
    the original MOT path so downstream FBX export uses the corrected motion.
    """
    resolved_mode = (mode or "off").lower()
    if resolved_mode not in {"off", "auto", "stance_only"}:
        raise ValueError(
            f"Unsupported post-ik foot snap mode: {mode}. "
            "Expected one of ['auto', 'off', 'stance_only']."
        )

    model_path = Path(model_path)
    mot_path = Path(mot_path)
    output_dir = Path(output_dir)
    contact_meta_path = Path(contact_meta_path)
    report_path = output_dir / f"{mot_path.stem}_foot_snap_report.json"
    report: Dict[str, Any] = {
        "mode": resolved_mode,
        "status": "disabled" if resolved_mode == "off" else "pending",
        "raw_mot_path": None,
        "corrected_mot_path": str(mot_path),
        "contact_meta_path": str(contact_meta_path),
        "target_clearance_m": float(target_clearance_m),
        "max_drop_m": float(max_drop_m),
        "transition_window": int(transition_window),
        "corrected_frames": 0,
        "stance_frames": 0,
        "max_applied_drop_m": 0.0,
        "mean_applied_drop_m": 0.0,
        "max_support_height_m": 0.0,
        "manual_plane_anchor_active": False,
        "scene_ground_used": False,
    }

    if resolved_mode == "off":
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    contact_meta = _read_contact_meta(contact_meta_path)
    report["manual_plane_anchor_active"] = bool(
        contact_meta.get("manual_plane_anchor_active", False)
    )
    report["scene_ground_used"] = bool(contact_meta.get("scene_ground_used", False))
    if (
        resolved_mode == "auto"
        and contact_meta.get("ground_alignment_applied_mode") != "contact_aware"
    ):
        report["status"] = "skipped"
        report["reason"] = "ground-alignment-not-contact-aware"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report
    if not contact_meta.get("available"):
        report["status"] = "skipped"
        report["reason"] = contact_meta.get("reason", "contact-meta-unavailable")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    header_lines, labels, data, _ = _read_mot(mot_path)
    if data.shape[0] == 0:
        report["status"] = "skipped"
        report["reason"] = "empty-mot"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    try:
        pelvis_ty_idx = labels.index("pelvis_ty")
    except ValueError:
        report["status"] = "skipped"
        report["reason"] = "pelvis_ty-missing"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    total_rows = data.shape[0]
    frame_count = min(
        data.shape[0],
        int(contact_meta.get("frame_count", data.shape[0])),
        len(contact_meta.get("any_contact", [])),
    )
    if frame_count <= 0:
        report["status"] = "skipped"
        report["reason"] = "no-contact-frames"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    if frame_count != total_rows:
        report["frame_count_trimmed"] = int(frame_count)

    trimmed_meta = dict(contact_meta)
    for key in ("left_contact", "right_contact", "any_contact", "flight"):
        trimmed_meta[key] = list(contact_meta.get(key, []))[:frame_count]
    trimmed_meta["frame_count"] = int(frame_count)

    any_contact = np.asarray(trimmed_meta["any_contact"], dtype=bool)
    report["stance_frames"] = int(np.sum(any_contact))
    if not np.any(any_contact):
        report["status"] = "skipped"
        report["reason"] = "no-stance-frames"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    marker_heights = _compute_marker_heights_from_mot(model_path, mot_path, FOOT_MARKERS)
    drop_series, diagnostics = _compute_stance_drop_series(
        marker_heights=marker_heights,
        contact_meta=trimmed_meta,
        target_clearance_m=float(target_clearance_m),
        max_drop_m=float(max_drop_m),
        transition_window=int(transition_window),
    )
    if drop_series.shape[0] != frame_count:
        report["status"] = "skipped"
        report["reason"] = diagnostics.get("reason", "drop-series-shape-mismatch")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    positive_mask = drop_series > 1e-5
    support_height = np.asarray(
        diagnostics.get("support_height", np.full(frame_count, np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    if not np.any(positive_mask):
        finite_support = support_height[np.isfinite(support_height)]
        report["status"] = "skipped"
        report["reason"] = "no-positive-drop"
        report["max_support_height_m"] = (
            float(np.max(finite_support)) if finite_support.size else 0.0
        )
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    raw_mot_path = mot_path.with_name(f"{mot_path.stem}_raw{mot_path.suffix}")
    shutil.copy2(mot_path, raw_mot_path)

    corrected = data.copy()
    corrected[:frame_count, pelvis_ty_idx] -= drop_series.astype(np.float64)
    _write_mot(mot_path, header_lines, labels, corrected)

    report.update(
        {
            "status": "applied",
            "raw_mot_path": str(raw_mot_path),
            "corrected_frames": int(np.sum(positive_mask)),
            "max_applied_drop_m": float(np.nanmax(drop_series)),
            "mean_applied_drop_m": float(np.nanmean(drop_series[positive_mask])),
            "max_support_height_m": float(np.nanmax(support_height[np.isfinite(support_height)]))
            if np.any(np.isfinite(support_height))
            else 0.0,
        }
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
