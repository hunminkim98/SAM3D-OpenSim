"""
SAM3D Body inference wrapper for video processing.

This module provides a high-level interface to run SAM3D Body inference
on video frames and extract MHR70 keypoints.
"""

import os
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm

from src.moge_scene_ground import (
    compute_output_scene_ground,
    estimate_ground_plane_from_scene_points,
)


class SelectionCancelledError(RuntimeError):
    """Raised when the user aborts the first-frame person selection."""


class SAM3DInference:
    """
    Wrapper for SAM3D Body model inference.

    Handles model loading, frame processing, and result extraction.
    """

    TRACKING_KEYPOINT_INDICES = tuple(list(range(21)) + list(range(63, 70)))

    @staticmethod
    def _format_vram_gb(total_memory_bytes: int) -> str:
        return f"{total_memory_bytes / (1024 ** 3):.1f} GB"

    @staticmethod
    def _normalize_optional_path(path: Optional[str]) -> str:
        """Convert nullable config paths to the empty-string convention used upstream."""
        if path is None:
            return ""
        return str(path)

    def _add_import_path(self, path: Path, label: str) -> bool:
        """Add an import root to sys.path once, if it exists."""
        if not path.exists():
            return False

        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        print(f"Using {label}: {path}")
        return True

    @staticmethod
    def _candidate_sam3_import_roots(path: Path) -> list[Path]:
        """Normalize either the SAM3 repo root or package root into import roots."""
        candidates: list[Path] = []

        if (path / "sam3" / "model_builder.py").exists():
            candidates.append(path)

        if (path / "model_builder.py").exists() and (path / "__init__.py").exists():
            candidates.append(path.parent)

        return candidates

    @staticmethod
    def _extract_bbox(output: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return a normalized bbox array, or None if the output has no usable box."""
        bbox = output.get("bbox")
        if bbox is None:
            return None

        bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox_array.size != 4:
            return None

        return bbox_array

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        return width * height

    @staticmethod
    def _bbox_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
        center_a = np.array(
            [(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5],
            dtype=np.float32,
        )
        center_b = np.array(
            [(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5],
            dtype=np.float32,
        )
        return float(np.linalg.norm(center_a - center_b))

    @classmethod
    def _bbox_iou(cls, box_a: np.ndarray, box_b: np.ndarray) -> float:
        inter_x1 = max(float(box_a[0]), float(box_b[0]))
        inter_y1 = max(float(box_a[1]), float(box_b[1]))
        inter_x2 = min(float(box_a[2]), float(box_b[2]))
        inter_y2 = min(float(box_a[3]), float(box_b[3]))

        inter_area = cls._bbox_area(np.array([inter_x1, inter_y1, inter_x2, inter_y2]))
        if inter_area <= 0:
            return 0.0

        union = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    def _match_output_index_to_bbox(
        self,
        outputs: List[Dict[str, Any]],
        reference_bbox: Optional[np.ndarray],
    ) -> Optional[int]:
        """Match the best output to a reference bbox using the standard tracking score."""
        if not outputs:
            return None

        if reference_bbox is None:
            return 0

        best_index = None
        best_score = None

        for index, output in enumerate(outputs):
            bbox = self._extract_bbox(output)
            if bbox is None:
                continue

            score = (
                self._bbox_iou(reference_bbox, bbox),
                -self._bbox_center_distance(reference_bbox, bbox),
                self._bbox_area(bbox),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        return 0 if best_index is None else best_index

    @staticmethod
    def _format_bbox(bbox: Optional[np.ndarray]) -> Optional[list[float]]:
        if bbox is None:
            return None
        return [round(float(value), 2) for value in bbox.tolist()]

    @staticmethod
    def _bbox_diagonal(bbox: np.ndarray) -> float:
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        return float(np.hypot(width, height))

    @staticmethod
    def _clamp_bbox_to_image(
        bbox: Optional[np.ndarray],
        image_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Clamp a bbox to image bounds and reject degenerate boxes."""
        if bbox is None:
            return None

        height, width = image_shape[:2]
        clamped = np.asarray(bbox, dtype=np.float32).copy()
        clamped[0] = np.clip(clamped[0], 0, width - 1)
        clamped[1] = np.clip(clamped[1], 0, height - 1)
        clamped[2] = np.clip(clamped[2], 0, width - 1)
        clamped[3] = np.clip(clamped[3], 0, height - 1)

        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            return None

        return clamped

    @classmethod
    def _extract_tracking_keypoints(
        cls,
        output: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """Return the stable subset of 2D keypoints used for bbox tracking."""
        keypoints = output.get("pred_keypoints_2d")
        if keypoints is None:
            return None

        points = np.asarray(keypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return None

        if points.shape[0] > max(cls.TRACKING_KEYPOINT_INDICES):
            points = points[list(cls.TRACKING_KEYPOINT_INDICES), :2]
        else:
            points = points[:, :2]

        return points

    def _estimate_bbox_from_keypoints(
        self,
        output: Dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Estimate a tracking bbox from stable 2D keypoints."""
        points = self._extract_tracking_keypoints(output)
        if points is None:
            return None

        height, width = image_shape[:2]
        valid_mask = np.isfinite(points).all(axis=1)
        if not np.any(valid_mask):
            return None

        points = points[valid_mask]
        bounds_mask = (
            (points[:, 0] >= -0.25 * width)
            & (points[:, 0] <= 1.25 * width)
            & (points[:, 1] >= -0.25 * height)
            & (points[:, 1] <= 1.25 * height)
        )
        points = points[bounds_mask]
        if len(points) < 4:
            return None

        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        bbox_width = max(float(max_xy[0] - min_xy[0]), 1.0)
        bbox_height = max(float(max_xy[1] - min_xy[1]), 1.0)
        pad_x = max(self.tracking_min_padding_px, bbox_width * self.tracking_bbox_padding)
        pad_y = max(self.tracking_min_padding_px, bbox_height * self.tracking_bbox_padding)

        bbox = np.array(
            [
                min_xy[0] - pad_x,
                min_xy[1] - pad_y,
                max_xy[0] + pad_x,
                max_xy[1] + pad_y,
            ],
            dtype=np.float32,
        )

        center_x = float((bbox[0] + bbox[2]) * 0.5)
        center_y = float((bbox[1] + bbox[3]) * 0.5)
        width_with_padding = max(float(bbox[2] - bbox[0]), self.tracking_min_box_size_px)
        height_with_padding = max(float(bbox[3] - bbox[1]), self.tracking_min_box_size_px)

        expanded_bbox = np.array(
            [
                center_x - width_with_padding * 0.5,
                center_y - height_with_padding * 0.5,
                center_x + width_with_padding * 0.5,
                center_y + height_with_padding * 0.5,
            ],
            dtype=np.float32,
        )
        return self._clamp_bbox_to_image(expanded_bbox, image_shape)

    def _is_plausible_tracked_bbox(
        self,
        bbox: Optional[np.ndarray],
    ) -> bool:
        """Reject abrupt bbox jumps that usually indicate tracking drift."""
        if bbox is None:
            return False

        if self.selected_bbox_prev is None:
            return True

        previous_bbox = self.selected_bbox_prev
        previous_area = max(self._bbox_area(previous_bbox), 1.0)
        current_area = max(self._bbox_area(bbox), 1.0)
        area_ratio = current_area / previous_area
        if area_ratio > self.tracking_max_area_ratio:
            return False
        if area_ratio < 1.0 / self.tracking_max_area_ratio:
            return False

        center_jump = self._bbox_center_distance(previous_bbox, bbox)
        previous_diag = max(self._bbox_diagonal(previous_bbox), 1.0)
        return center_jump <= previous_diag * self.tracking_max_center_jump_ratio

    def _next_tracking_bbox_from_tracked_output(
        self,
        output: Dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Estimate the next bbox for the single-person fast path."""
        candidate_bbox = self._estimate_bbox_from_keypoints(output, image_shape)
        if candidate_bbox is None:
            candidate_bbox = self._clamp_bbox_to_image(
                self._extract_bbox(output),
                image_shape,
            )

        if not self._is_plausible_tracked_bbox(candidate_bbox):
            return None

        return candidate_bbox

    def _next_tracking_bbox_from_refresh_output(
        self,
        output: Dict[str, Any],
        image_shape: tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """Use a detector-derived bbox when available after a full refresh."""
        candidate_bbox = self._clamp_bbox_to_image(
            self._extract_bbox(output),
            image_shape,
        )
        if candidate_bbox is not None:
            return candidate_bbox

        return self._estimate_bbox_from_keypoints(output, image_shape)

    def _should_run_full_refresh(self, frame_idx: int) -> bool:
        """Decide when to rerun full multi-person detection."""
        if self.selected_bbox_prev is None:
            return True

        if self.tracking_consecutive_failures >= self.tracking_max_consecutive_failures:
            return True

        if self.last_full_detection_frame is None:
            return True

        return (
            frame_idx - self.last_full_detection_frame
            >= self.tracking_refresh_interval_frames
        )

    @staticmethod
    def _prepare_preview_base(
        image: np.ndarray,
        max_width: int = 1600,
        max_height: int = 900,
    ) -> Tuple[np.ndarray, float]:
        """Resize a preview image to a reasonable interactive window size."""
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            import cv2

            preview_base = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview_base = image.copy()
        return preview_base, scale

    @staticmethod
    def _normalize_drag_roi(
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        image_shape: tuple[int, ...],
        min_size_px: float = 12.0,
    ) -> Optional[np.ndarray]:
        """Convert two drag points into a valid ROI rectangle."""
        height, width = image_shape[:2]
        x1 = float(np.clip(min(start_xy[0], end_xy[0]), 0, width - 1))
        y1 = float(np.clip(min(start_xy[1], end_xy[1]), 0, height - 1))
        x2 = float(np.clip(max(start_xy[0], end_xy[0]), 0, width - 1))
        y2 = float(np.clip(max(start_xy[1], end_xy[1]), 0, height - 1))
        if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _resize_mask_for_preview(mask: np.ndarray, preview_shape: tuple[int, ...]) -> np.ndarray:
        """Resize a boolean mask to the preview resolution."""
        import cv2

        preview_height, preview_width = preview_shape[:2]
        resized = cv2.resize(
            mask.astype(np.uint8),
            (preview_width, preview_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)

    def _build_support_surface_preview(
        self,
        scene_output: Dict[str, np.ndarray],
        image_shape: tuple[int, ...],
        roi_xyxy: np.ndarray,
        exclude_bboxes: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Fit a plane inside the user ROI and prepare preview metadata."""
        plane_model = estimate_ground_plane_from_scene_points(
            scene_output.get("points"),
            scene_output.get("mask"),
            image_shape,
            exclude_bboxes=exclude_bboxes,
            roi_xyxy=roi_xyxy,
            mode="manual_roi",
        )
        if plane_model is None:
            return {
                "fit_valid": False,
                "roi_xyxy": self._format_bbox(roi_xyxy),
                "warning": "Plane fit weak. Press R to redraw or S to skip.",
            }

        plane_model["fit_valid"] = True
        plane_model["warning"] = None
        return plane_model

    def _render_support_surface_overlay(
        self,
        image: np.ndarray,
        selection_state: Dict[str, Any],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Render the manual support-surface ROI and preview gate."""
        import cv2

        preview = image.copy()
        overlay = preview.copy()

        current_roi = selection_state.get("roi_xyxy")
        if selection_state.get("drag_start") is not None and selection_state.get("drag_current") is not None:
            drag_roi = self._normalize_drag_roi(
                selection_state["drag_start"],
                selection_state["drag_current"],
                selection_state.get("source_image_shape", preview.shape),
            )
            if drag_roi is not None:
                current_roi = drag_roi

        plane_preview = selection_state.get("plane_preview") or {}
        inlier_mask = plane_preview.get("inlier_mask_debug")
        candidate_mask = plane_preview.get("candidate_mask_debug")
        if candidate_mask is not None:
            candidate_preview = self._resize_mask_for_preview(candidate_mask, preview.shape)
            overlay[candidate_preview] = (
                overlay[candidate_preview] * 0.55
                + np.array([255, 200, 0], dtype=np.float32) * 0.45
            )
        if inlier_mask is not None:
            inlier_preview = self._resize_mask_for_preview(inlier_mask, preview.shape)
            overlay[inlier_preview] = (
                overlay[inlier_preview] * 0.35
                + np.array([0, 220, 120], dtype=np.float32) * 0.65
            )
        if candidate_mask is not None or inlier_mask is not None:
            preview = overlay.astype(np.uint8)

        if current_roi is not None:
            x1, y1, x2, y2 = (np.asarray(current_roi, dtype=np.float32) * scale).astype(int)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 220), 2)

        title = "Select support surface ROI"
        if selection_state.get("ui_state") == "preview":
            title = "Preview support plane before accept"
        cv2.putText(
            preview,
            title,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        instructions = [
            "Draw one ROI over the treadmill deck / box top / contact surface.",
            "Preview is mandatory before accept.",
            "Enter: accept preview  R: redraw  S: skip to auto  Esc: fallback to auto",
        ]
        text_y = 72
        for line in instructions:
            cv2.putText(
                preview,
                line,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_y += 26

        status_lines = [
            f"State: {selection_state.get('ui_state', 'drawing')}",
            f"Preview shown: {'yes' if selection_state.get('preview_shown') else 'no'}",
            f"Accept enabled: {'yes' if selection_state.get('fit_valid') else 'no'}",
        ]
        if current_roi is not None:
            roi_label = self._format_bbox(np.asarray(current_roi, dtype=np.float32))
            status_lines.append(f"ROI: {roi_label}")

        if plane_preview:
            confidence = plane_preview.get("confidence")
            inlier_ratio = plane_preview.get("inlier_ratio")
            num_candidates = plane_preview.get("num_candidates")
            if confidence is not None:
                status_lines.append(f"Plane confidence: {float(confidence):.2f}")
            if inlier_ratio is not None:
                status_lines.append(f"Inlier ratio: {float(inlier_ratio):.2f}")
            if num_candidates is not None:
                status_lines.append(f"Valid scene points: {int(num_candidates)}")
            if plane_preview.get("source_mode"):
                status_lines.append(f"Mode: {plane_preview['source_mode']}")

        warning = plane_preview.get("warning")
        if warning:
            status_lines.append(f"Warning: {warning}")

        line_y = text_y + 10
        line_color = (255, 255, 255)
        if warning:
            line_color = (0, 180, 255)
        for line in status_lines:
            cv2.putText(
                preview,
                line,
                (20, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                line_color,
                2,
                cv2.LINE_AA,
            )
            line_y += 24

        return preview

    def _select_support_surface_with_gui(
        self,
        image: np.ndarray,
        scene_output: Dict[str, np.ndarray],
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Require ROI preview acceptance before finalizing a support plane."""
        import cv2

        window_name = f"Select support surface - frame {frame_idx}"
        preview_base, scale = self._prepare_preview_base(image)
        exclude_bboxes = [
            bbox
            for bbox in (self._extract_bbox(output) for output in outputs)
            if bbox is not None
        ]
        selection_state: Dict[str, Any] = {
            "source_image_shape": image.shape,
            "drag_start": None,
            "drag_current": None,
            "roi_xyxy": None,
            "plane_preview": None,
            "fit_valid": False,
            "preview_shown": False,
            "accepted": False,
            "ui_state": "drawing",
        }

        def _reset_to_drawing() -> None:
            selection_state["drag_start"] = None
            selection_state["drag_current"] = None
            selection_state["roi_xyxy"] = None
            selection_state["plane_preview"] = None
            selection_state["fit_valid"] = False
            selection_state["preview_shown"] = False
            selection_state["accepted"] = False
            selection_state["ui_state"] = "drawing"

        def _finalize_preview() -> None:
            roi_xyxy = selection_state.get("roi_xyxy")
            if roi_xyxy is None:
                selection_state["plane_preview"] = {
                    "fit_valid": False,
                    "warning": "ROI too small. Press R to redraw or S to skip.",
                }
                selection_state["fit_valid"] = False
                selection_state["preview_shown"] = True
                selection_state["ui_state"] = "preview"
                return

            plane_preview = self._build_support_surface_preview(
                scene_output=scene_output,
                image_shape=image.shape,
                roi_xyxy=np.asarray(roi_xyxy, dtype=np.float32),
                exclude_bboxes=exclude_bboxes,
            )
            selection_state["plane_preview"] = plane_preview
            selection_state["fit_valid"] = bool(plane_preview.get("fit_valid", False))
            selection_state["preview_shown"] = True
            selection_state["ui_state"] = "preview"

        def handle_mouse(event, x, y, _flags, _userdata):
            if selection_state["ui_state"] != "drawing":
                return

            point_x = float(x) / scale
            point_y = float(y) / scale

            if event == cv2.EVENT_LBUTTONDOWN:
                selection_state["drag_start"] = (point_x, point_y)
                selection_state["drag_current"] = (point_x, point_y)
            elif event == cv2.EVENT_MOUSEMOVE and selection_state["drag_start"] is not None:
                selection_state["drag_current"] = (point_x, point_y)
            elif event == cv2.EVENT_LBUTTONUP and selection_state["drag_start"] is not None:
                selection_state["drag_current"] = (point_x, point_y)
                roi_xyxy = self._normalize_drag_roi(
                    selection_state["drag_start"],
                    selection_state["drag_current"],
                    image.shape,
                )
                selection_state["roi_xyxy"] = roi_xyxy
                selection_state["drag_start"] = None
                selection_state["drag_current"] = None
                _finalize_preview()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, handle_mouse)

        try:
            while True:
                overlay = self._render_support_surface_overlay(
                    preview_base,
                    selection_state,
                    scale=scale,
                )
                cv2.imshow(window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(50) & 0xFF

                if key == 27:
                    return None, "fallback-auto"
                if key in (ord("s"), ord("S")):
                    return None, "skipped"
                if key in (ord("r"), ord("R")):
                    _reset_to_drawing()
                    continue
                if key in (10, 13):
                    if selection_state["ui_state"] == "preview" and selection_state["fit_valid"]:
                        selection_state["accepted"] = True
                        return selection_state["plane_preview"], "manual-roi"
        finally:
            cv2.destroyWindow(window_name)

    def _render_selection_overlay(
        self,
        image: np.ndarray,
        outputs: List[Dict[str, Any]],
        selected_index: Optional[int],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Draw numbered person boxes and instructions on a preview image."""
        import cv2

        preview = image.copy()
        default_color = (255, 170, 0)
        selected_color = (0, 220, 0)

        for index, output in enumerate(outputs):
            bbox = self._extract_bbox(output)
            if bbox is None:
                continue

            x1, y1, x2, y2 = (bbox * scale).astype(int)
            color = selected_color if index == selected_index else default_color
            thickness = 4 if index == selected_index else 2
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)

            label = f"Person {index}"
            text_scale = 0.8
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_thickness,
            )
            box_top = max(0, y1 - text_height - baseline - 10)
            cv2.rectangle(
                preview,
                (x1, box_top),
                (x1 + text_width + 12, box_top + text_height + baseline + 10),
                color,
                -1,
            )
            cv2.putText(
                preview,
                label,
                (x1 + 6, box_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA,
            )

        instructions = [
            "Select one person for tracking.",
            "Click a box or press a digit key, then press Enter to confirm.",
            "Press Esc to cancel.",
        ]
        text_origin_y = 36
        for line in instructions:
            cv2.putText(
                preview,
                line,
                (20, text_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_origin_y += 30

        if selected_index is not None:
            status = f"Selected: Person {selected_index}"
            cv2.putText(
                preview,
                status,
                (20, text_origin_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                selected_color,
                2,
                cv2.LINE_AA,
            )

        return preview

    def _select_person_with_gui(
        self,
        image: np.ndarray,
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[int, str]:
        """Open a preview window and let the user choose one detected person."""
        import cv2

        window_name = f"Select tracked person - frame {frame_idx}"
        max_width = 1600
        max_height = 900
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)

        if scale < 1.0:
            preview_base = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview_base = image.copy()

        selection_state = {"selected_index": None}

        def handle_mouse(event, x, y, _flags, _userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            point_x = float(x) / scale
            point_y = float(y) / scale
            for index, output in enumerate(outputs):
                bbox = self._extract_bbox(output)
                if bbox is None:
                    continue
                if bbox[0] <= point_x <= bbox[2] and bbox[1] <= point_y <= bbox[3]:
                    selection_state["selected_index"] = index
                    break

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, handle_mouse)

        try:
            while True:
                overlay = self._render_selection_overlay(
                    preview_base,
                    outputs,
                    selection_state["selected_index"],
                    scale=scale,
                )
                cv2.imshow(window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(50) & 0xFF

                if key == 27:
                    raise SelectionCancelledError("Person selection cancelled by user.")

                if ord("0") <= key <= ord("9"):
                    digit = key - ord("0")
                    if digit < len(outputs):
                        selection_state["selected_index"] = digit

                if key in (10, 13) and selection_state["selected_index"] is not None:
                    return selection_state["selected_index"], "interactive"
        finally:
            cv2.destroyWindow(window_name)

    def _select_person_with_cli(
        self,
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[int, str]:
        """Fallback terminal selection when OpenCV UI is unavailable."""
        print(f"Multiple people detected in frame {frame_idx}.")
        for index, output in enumerate(outputs):
            bbox = self._extract_bbox(output)
            bbox_desc = self._format_bbox(bbox)
            area_desc = round(self._bbox_area(bbox), 1) if bbox is not None else "n/a"
            print(f"  [{index}] bbox={bbox_desc} area={area_desc}")

        while True:
            try:
                raw_value = input(f"Select person index [0-{len(outputs) - 1}]: ").strip()
            except EOFError as exc:
                raise SelectionCancelledError(
                    "Person selection requires interactive input."
                ) from exc

            if not raw_value:
                continue

            try:
                selected_index = int(raw_value)
            except ValueError:
                print("Invalid selection. Enter the numeric person index.")
                continue

            if 0 <= selected_index < len(outputs):
                return selected_index, "fallback-cli"

            print(f"Selection must be between 0 and {len(outputs) - 1}.")

    def _initialize_selected_person(
        self,
        image: np.ndarray,
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Optional[int]:
        """Choose the first tracked person once and keep the choice for later frames."""
        if not outputs:
            return None

        if len(outputs) == 1:
            selected_index = 0
            selection_mode = "auto-single-detection"
        else:
            print(
                f"Detected {len(outputs)} people in frame {frame_idx}. "
                "Select the person to track."
            )
            try:
                selected_index, selection_mode = self._select_person_with_gui(
                    image=image,
                    outputs=outputs,
                    frame_idx=frame_idx,
                )
            except SelectionCancelledError:
                raise
            except Exception as exc:
                print(f"OpenCV selection UI unavailable, falling back to terminal input: {exc}")
                selected_index, selection_mode = self._select_person_with_cli(
                    outputs=outputs,
                    frame_idx=frame_idx,
                )

        selected_bbox = self._extract_bbox(outputs[selected_index])
        self.selected_person_initialized = True
        self.selected_person_index_first_frame = selected_index
        self.selected_first_frame_index = frame_idx
        self.selected_bbox_prev = None if selected_bbox is None else selected_bbox.copy()
        self.selected_bbox_first_frame = self._format_bbox(selected_bbox)
        self.selection_mode = selection_mode
        self.last_full_detection_frame = frame_idx
        self.tracking_consecutive_failures = 0

        print(
            f"Tracking person {selected_index} from frame {frame_idx} "
            f"({selection_mode})."
        )
        return selected_index

    def _match_selected_output_index(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Optional[int]:
        """Match the previously selected person to the current frame."""
        return self._match_output_index_to_bbox(outputs, self.selected_bbox_prev)

    def _prepare_component_paths(
        self,
        detector_name: Optional[str],
        detector_path: Optional[str],
        fov_name: Optional[str],
        fov_path: Optional[str],
    ) -> Tuple[str, str]:
        """Resolve optional dependency paths before importing SAM3D helpers."""
        normalized_detector_path = self._normalize_optional_path(detector_path)
        normalized_fov_path = self._normalize_optional_path(fov_path)

        if detector_name == "sam3":
            candidate_paths = []
            if normalized_detector_path:
                detector_root = Path(normalized_detector_path)
                candidate_paths.extend(self._candidate_sam3_import_roots(detector_root))

            default_root = self.sam3d_root.parent / "sam3"
            candidate_paths.extend(self._candidate_sam3_import_roots(default_root))
            candidate_paths.extend(self._candidate_sam3_import_roots(default_root / "sam3"))

            for candidate in candidate_paths:
                if self._add_import_path(candidate, "SAM3 import path"):
                    break
            else:
                print(
                    "Warning: Could not locate a SAM3 import path. "
                    "Expected something like C:/Sam3dBody/sam3/sam3."
                )

        if fov_name == "moge2" and not normalized_fov_path:
            print("Using MoGe2 default pretrained model: Ruicheng/moge-2-vitl-normal")

        return normalized_detector_path, normalized_fov_path

    def __init__(
        self,
        sam3d_root: str = "C:/Sam3dBody/sam-3d-body",
        checkpoint_path: str = None,
        mhr_path: str = None,
        device: str = "cuda",
        detector_name: str = "vitdet",
        detector_path: str = None,
        segmentor_name: str = None,
        segmentor_path: str = None,
        fov_name: str = None,
        fov_path: str = None,
        bbox_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
        single_person: bool = True,
        support_surface_mode: str = "auto",
    ):
        """
        Initialize SAM3D Body inference.

        Args:
            sam3d_root: Path to SAM3D Body installation
            checkpoint_path: Path to model checkpoint
            mhr_path: Path to MHR model assets
            device: Device for inference ('cuda' or 'cpu')
            detector_name: Human detector name ('vitdet' or None)
            detector_path: Path to detector model
            segmentor_name: Segmentor name ('sam2' or None)
            segmentor_path: Path to segmentor model
            fov_name: FOV estimator name ('moge2' or None)
            fov_path: Path to FOV estimator model
            bbox_threshold: Detection threshold
            nms_threshold: NMS threshold
            use_mask: Whether to use segmentation masks
            inference_type: 'full', 'body', or 'hand'
            single_person: Whether to select and track one person across the clip
            support_surface_mode: Scene-ground support-surface selection mode
        """
        self.sam3d_root = Path(sam3d_root)
        self.requested_device = str(device).lower()
        self.cuda_available = torch.cuda.is_available()

        if self.requested_device == "cuda" and self.cuda_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.bbox_threshold = bbox_threshold
        self.nms_threshold = nms_threshold
        self.use_mask = use_mask
        self.inference_type = inference_type
        self.single_person = bool(single_person)
        self.per_frame_logging = False
        self.selected_person_initialized = False
        self.selected_person_index_first_frame = None
        self.selected_first_frame_index = None
        self.selected_bbox_first_frame = None
        self.selected_bbox_prev = None
        self.selection_mode = "disabled" if not self.single_person else None
        self.tracking_refresh_interval_frames = 30
        self.tracking_max_consecutive_failures = 2
        self.tracking_bbox_padding = 0.25
        self.tracking_min_padding_px = 24.0
        self.tracking_min_box_size_px = 96.0
        self.tracking_max_area_ratio = 3.5
        self.tracking_max_center_jump_ratio = 1.75
        self.last_full_detection_frame = None
        self.tracking_consecutive_failures = 0
        self.tracking_fast_path_frames = 0
        self.tracking_refresh_frames = 0
        self.tracking_recovery_events = 0
        self.scene_ground_enabled = bool(self.single_person and fov_name == "moge2")
        self.support_surface_mode_requested = str(support_surface_mode or "auto")
        self.support_surface_mode_applied = "auto"
        self.support_surface_selection_status = (
            "auto" if self.support_surface_mode_requested == "auto" else "pending"
        )
        self.support_surface_manual_confirmed = False
        self.support_surface_source_frame_idx = None
        self.support_surface_roi_xyxy = None
        self.scene_ground_model: Optional[Dict[str, Any]] = None
        self.scene_ground_attempted_frames: set[int] = set()
        self.scene_ground_available_frames = 0
        self.scene_fov_estimator = None
        self.mesh_visualization_fov_estimator = None

        if self.support_surface_mode_requested == "manual_roi" and not self.scene_ground_enabled:
            print(
                "Warning: manual support-surface selection requires "
                "single_person=true and fov=moge2. Falling back to auto."
            )
            self.support_surface_selection_status = "fallback-auto"

        self._log_device_info()

        # Add SAM3D Body to path
        if str(self.sam3d_root) not in sys.path:
            sys.path.insert(0, str(self.sam3d_root))

        detector_path, fov_path = self._prepare_component_paths(
            detector_name=detector_name,
            detector_path=detector_path,
            fov_name=fov_name,
            fov_path=fov_path,
        )

        # Set default paths
        if checkpoint_path is None:
            checkpoint_path = str(
                self.sam3d_root.parent
                / "checkpoints"
                / "sam-3d-body-dinov3"
                / "model.ckpt"
            )
        if mhr_path is None:
            mhr_path = str(
                self.sam3d_root.parent
                / "checkpoints"
                / "sam-3d-body-dinov3"
                / "assets"
                / "mhr_model.pt"
            )

        # Load model
        self._load_model(
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            detector_name=detector_name,
            detector_path=detector_path,
            segmentor_name=segmentor_name,
            segmentor_path=segmentor_path,
            fov_name=fov_name,
            fov_path=fov_path,
        )

        # Cache for focal length (computed on first frame)
        self._cached_focal_length = None

    def _log_device_info(self):
        """Print the requested and resolved inference device."""
        print(f"Requested inference device: {self.requested_device}")

        if self.device.type == "cuda":
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()

            try:
                device_props = torch.cuda.get_device_properties(device_index)
                print(
                    "Resolved inference device: "
                    f"cuda:{device_index} ({device_props.name}, "
                    f"{self._format_vram_gb(device_props.total_memory)} VRAM)"
                )
            except Exception:
                print(f"Resolved inference device: cuda:{device_index}")

            print(f"CUDA device count: {torch.cuda.device_count()}")
            return

        print("Resolved inference device: cpu")
        if self.requested_device == "cuda" and not self.cuda_available:
            print("Device note: CUDA is not available, falling back to CPU.")
        else:
            print("Device note: CPU inference requested.")

    def _load_model(
        self,
        checkpoint_path: str,
        mhr_path: str,
        detector_name: str,
        detector_path: str,
        segmentor_name: str,
        segmentor_path: str,
        fov_name: str,
        fov_path: str,
    ):
        """Load SAM3D Body model and optional components."""
        # Import SAM3D Body modules
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

        print(f"Loading SAM3D Body model from {checkpoint_path}")
        self.model, self.model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=str(self.device),
            mhr_path=mhr_path,
        )

        # Initialize optional components
        human_detector = None
        human_segmentor = None
        fov_estimator = None

        if detector_name:
            try:
                from tools.build_detector import HumanDetector

                human_detector = HumanDetector(
                    name=detector_name,
                    device=self.device,
                    path=detector_path,
                )
                print(f"Loaded human detector: {detector_name}")
            except Exception as e:
                print(f"Warning: Could not load detector: {e}")

        if segmentor_name:
            try:
                from tools.build_sam import HumanSegmentor

                human_segmentor = HumanSegmentor(
                    name=segmentor_name,
                    device=self.device,
                    path=segmentor_path,
                )
                print(f"Loaded segmentor: {segmentor_name}")
            except Exception as e:
                print(f"Warning: Could not load segmentor: {e}")

        if fov_name:
            try:
                from tools.build_fov_estimator import FOVEstimator

                fov_estimator = FOVEstimator(
                    name=fov_name,
                    device=self.device,
                    path=fov_path,
                )
                print(f"Loaded FOV estimator: {fov_name}")
                self.mesh_visualization_fov_estimator = fov_estimator
                if fov_name == "moge2":
                    self.scene_fov_estimator = fov_estimator
            except Exception as e:
                print(f"Warning: Could not load FOV estimator: {e}")

        # Create estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=human_detector,
            human_segmentor=human_segmentor,
            fov_estimator=fov_estimator,
        )

        # Get mesh faces for export
        self.faces = self.estimator.faces if hasattr(self.estimator, "faces") else None

        print("SAM3D Body model loaded successfully")

    def _infer_moge_scene(self, image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Run full MoGe inference for scene geometry using the already-loaded model."""
        if self.scene_fov_estimator is None:
            return None

        moge_model = getattr(self.scene_fov_estimator, "fov_estimator", None)
        if moge_model is None:
            return None

        input_tensor = torch.tensor(
            image / 255.0,
            dtype=torch.float32,
            device=self.device,
        ).permute(2, 0, 1)

        with torch.inference_mode():
            scene_output = moge_model.infer(input_tensor)

        return {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in scene_output.items()
        }

    def _ensure_scene_ground_model(
        self,
        image: np.ndarray,
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> None:
        """Estimate the clip-level ground plane once from MoGe scene geometry."""
        if not self.scene_ground_enabled or self.scene_ground_model is not None:
            return
        if frame_idx in self.scene_ground_attempted_frames:
            return

        self.scene_ground_attempted_frames.add(frame_idx)
        scene_output = self._infer_moge_scene(image)
        if not scene_output:
            if self.support_surface_mode_requested == "manual_roi":
                self.support_surface_mode_applied = "auto"
                self.support_surface_selection_status = "fallback-auto"
            return

        candidate_bboxes = [
            bbox
            for bbox in (self._extract_bbox(output) for output in outputs)
            if bbox is not None
        ]

        plane_model = None
        if self.support_surface_mode_requested == "manual_roi":
            print(
                "Manual support-surface mode enabled. "
                "Draw an ROI and accept the previewed support plane to continue."
            )
            try:
                manual_plane, selection_status = self._select_support_surface_with_gui(
                    image=image,
                    scene_output=scene_output,
                    outputs=outputs,
                    frame_idx=frame_idx,
                )
            except Exception as exc:
                print(
                    "Warning: Support-surface ROI UI unavailable, "
                    f"falling back to auto plane estimation: {exc}"
                )
                manual_plane = None
                selection_status = "fallback-auto"

            if manual_plane is not None:
                plane_model = manual_plane
                self.support_surface_mode_applied = "manual_roi"
                self.support_surface_selection_status = "accepted"
                self.support_surface_manual_confirmed = True
                self.support_surface_source_frame_idx = int(frame_idx)
                self.support_surface_roi_xyxy = plane_model.get("roi_xyxy")
                print(
                    "Manual support surface accepted on frame "
                    f"{frame_idx}."
                )
            else:
                self.support_surface_mode_applied = "auto"
                self.support_surface_selection_status = selection_status
                self.support_surface_manual_confirmed = False
                self.support_surface_source_frame_idx = int(frame_idx)
                self.support_surface_roi_xyxy = None
                if selection_status == "skipped":
                    print("Support surface selection skipped, falling back to auto.")
                elif selection_status == "fallback-auto":
                    print("Support surface selection cancelled, falling back to auto.")

        if plane_model is None:
            plane_model = estimate_ground_plane_from_scene_points(
                scene_output.get("points"),
                scene_output.get("mask"),
                image.shape,
                exclude_bboxes=candidate_bboxes,
                mode="auto",
            )
            if self.support_surface_mode_requested == "auto":
                self.support_surface_mode_applied = "auto"
                self.support_surface_selection_status = "auto"

        if plane_model is None:
            print(
                f"Warning: Could not estimate MoGe ground plane from frame {frame_idx}."
            )
            return

        plane_model["frame_idx"] = int(frame_idx)
        self.scene_ground_model = plane_model
        confidence = plane_model.get("confidence", 0.0)
        inlier_ratio = plane_model.get("inlier_ratio", 0.0)
        source_mode = plane_model.get("source_mode", "unknown")
        if source_mode == "manual-roi":
            print(
                "Estimated MoGe support plane from accepted ROI on frame "
                f"{frame_idx} (confidence={confidence:.2f}, "
                f"inlier_ratio={inlier_ratio:.2f})"
            )
        else:
            print(
                "Estimated MoGe ground plane from frame "
                f"{frame_idx} (confidence={confidence:.2f}, "
                f"inlier_ratio={inlier_ratio:.2f})"
            )

    def _attach_scene_ground_to_output(self, output: Optional[Dict[str, Any]]) -> None:
        """Annotate a selected output with per-frame foot clearance metadata."""
        if output is None or not self.scene_ground_enabled or self.scene_ground_model is None:
            return

        scene_ground = compute_output_scene_ground(output, self.scene_ground_model)
        if scene_ground is None:
            return

        output["scene_ground"] = scene_ground
        self.scene_ground_available_frames += 1

    def process_frame(
        self,
        image: np.ndarray,
        frame_idx: int = 0,
        bboxes: Optional[np.ndarray] = None,
        *,
        use_cached_focal_length: bool = True,
        force_fov_estimator=None,
    ) -> List[Dict[str, Any]]:
        """
        Process a single frame and extract 3D pose.

        Args:
            image: Input image as numpy array (RGB format)
            frame_idx: Frame index (used for focal length caching)
            bboxes: Optional pre-computed bounding boxes [N, 4]

        Returns:
            List of dictionaries, one per detected person, containing:
                - pred_keypoints_3d: (70, 3) MHR70 3D keypoints
                - pred_keypoints_2d: (70, 3) 2D projections with confidence
                - pred_vertices: (6890, 3) mesh vertices
                - pred_pose_raw: (72,) pose parameters
                - shape_params: (10,) shape parameters
                - pred_cam_t: (3,) camera translation
                - focal_length: camera focal length
                - bbox: (4,) bounding box
        """
        # Process image
        original_fov_estimator = None
        if force_fov_estimator is not None and hasattr(self.estimator, "fov_estimator"):
            original_fov_estimator = self.estimator.fov_estimator
            self.estimator.fov_estimator = force_fov_estimator

        try:
            if self.per_frame_logging:
                outputs = self.estimator.process_one_image(
                    image,
                    bboxes=bboxes,
                    bbox_thr=self.bbox_threshold,
                    nms_thr=self.nms_threshold,
                    use_mask=self.use_mask,
                    inference_type=self.inference_type,
                )
            else:
                with redirect_stdout(io.StringIO()):
                    outputs = self.estimator.process_one_image(
                        image,
                        bboxes=bboxes,
                        bbox_thr=self.bbox_threshold,
                        nms_thr=self.nms_threshold,
                        use_mask=self.use_mask,
                        inference_type=self.inference_type,
                    )
        finally:
            if force_fov_estimator is not None and hasattr(self.estimator, "fov_estimator"):
                self.estimator.fov_estimator = original_fov_estimator

        # Cache focal length from the first successful detection only once.
        if use_cached_focal_length and self._cached_focal_length is None and outputs:
            if "focal_length" in outputs[0]:
                self._cached_focal_length = outputs[0]["focal_length"]
            # Disable FOV estimator for subsequent frames (expensive)
            if (
                hasattr(self.estimator, "fov_estimator")
                and self.estimator.fov_estimator is not None
            ):
                self.estimator.fov_estimator = None
        elif use_cached_focal_length and self._cached_focal_length is not None:
            # Apply cached focal length to subsequent frames
            for out in outputs:
                out["focal_length"] = self._cached_focal_length

        return outputs

    def process_video_for_mesh_sidecars(
        self,
        frame_paths: List[str],
        progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a visualization-quality pass for mesh sidecar generation.

        This intentionally bypasses the tracked-bbox fast path and focal-length
        caching so the rendered mesh more closely matches the official upstream
        visualization semantics.
        """
        import cv2

        mesh_outputs = []
        iterator = tqdm(frame_paths, desc="Processing mesh sidecar frames") if progress else frame_paths

        reference_bbox = None
        if self.selected_bbox_first_frame is not None:
            reference_bbox = np.asarray(self.selected_bbox_first_frame, dtype=np.float32)

        full_refresh_fov_estimator = getattr(self, "mesh_visualization_fov_estimator", None)

        for idx, frame_path in enumerate(iterator):
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Warning: Could not read {frame_path} for mesh visualization")
                mesh_outputs.append(
                    {
                        "frame_idx": idx,
                        "frame_path": frame_path,
                        "outputs": [],
                        "output": None,
                        "selected_output_index": None,
                        "people_count": 0,
                        "inference_mode": "mesh_full_refresh",
                    }
                )
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            outputs = self.process_frame(
                image,
                frame_idx=idx,
                bboxes=None,
                use_cached_focal_length=False,
                force_fov_estimator=full_refresh_fov_estimator,
            )

            selected_output = None
            selected_output_index = None
            if self.single_person:
                selected_output_index = self._match_output_index_to_bbox(outputs, reference_bbox)
                if (
                    selected_output_index is None
                    and self.selected_person_index_first_frame is not None
                    and self.selected_person_index_first_frame < len(outputs)
                ):
                    selected_output_index = self.selected_person_index_first_frame

                if selected_output_index is not None and len(outputs) > selected_output_index:
                    selected_output = outputs[selected_output_index]
                    next_bbox = self._extract_bbox(selected_output)
                    reference_bbox = None if next_bbox is None else next_bbox.copy()
            else:
                selected_output = outputs[0] if outputs else None
                selected_output_index = 0 if outputs else None

            mesh_outputs.append(
                {
                    "frame_idx": idx,
                    "frame_path": frame_path,
                    "outputs": outputs,
                    "output": selected_output,
                    "selected_output_index": selected_output_index,
                    "people_count": len(outputs),
                    "inference_mode": "mesh_full_refresh",
                }
            )

        return mesh_outputs

    def get_selection_metadata(self) -> Dict[str, Any]:
        """Return metadata describing how the tracked person was chosen."""
        return {
            "single_person": self.single_person,
            "selection_mode": self.selection_mode,
            "selected_person_index_first_frame": self.selected_person_index_first_frame,
            "selected_first_frame_index": self.selected_first_frame_index,
            "selected_bbox_first_frame": self.selected_bbox_first_frame,
            "tracking_refresh_interval_frames": self.tracking_refresh_interval_frames,
            "tracking_fast_path_frames": self.tracking_fast_path_frames,
            "tracking_refresh_frames": self.tracking_refresh_frames,
            "tracking_recovery_events": self.tracking_recovery_events,
            "support_surface_mode_requested": self.support_surface_mode_requested,
        }

    def get_scene_ground_metadata(self) -> Dict[str, Any]:
        """Return clip-level metadata for MoGe-assisted scene grounding."""
        plane = self.scene_ground_model or {}
        normal = plane.get("normal_cam")
        if isinstance(normal, np.ndarray):
            normal = normal.tolist()
        return {
            "enabled": self.scene_ground_enabled,
            "available": self.scene_ground_model is not None,
            "source_frame_idx": plane.get("frame_idx"),
            "normal_cam": normal,
            "offset_cam": plane.get("offset_cam"),
            "confidence": plane.get("confidence"),
            "inlier_ratio": plane.get("inlier_ratio"),
            "residual_median": plane.get("residual_median"),
            "available_frames": self.scene_ground_available_frames,
            "support_surface_mode_requested": self.support_surface_mode_requested,
            "support_surface_mode_applied": self.support_surface_mode_applied,
            "support_surface_selection_status": self.support_surface_selection_status,
            "support_surface_manual_confirmed": self.support_surface_manual_confirmed,
            "support_surface_source_frame_idx": self.support_surface_source_frame_idx,
            "support_surface_roi_xyxy": self.support_surface_roi_xyxy,
            "source_mode": plane.get("source_mode"),
        }

    def _resolve_selected_output_from_full_detection(
        self,
        image: np.ndarray,
        outputs: List[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[np.ndarray]]:
        """Pick the tracked identity from a full multi-person detection frame."""
        if not outputs:
            return None, None, None

        if not self.selected_person_initialized:
            selected_output_index = self._initialize_selected_person(
                image=image,
                outputs=outputs,
                frame_idx=frame_idx,
            )
        else:
            selected_output_index = self._match_selected_output_index(outputs)

        if selected_output_index is None:
            return None, None, None

        selected_output = outputs[selected_output_index]
        next_bbox = self._next_tracking_bbox_from_refresh_output(
            selected_output,
            image.shape,
        )
        return selected_output, selected_output_index, next_bbox

    def process_video(
        self,
        frame_paths: List[str],
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all frames from a video.

        Args:
            frame_paths: List of paths to frame images
            progress: Whether to show progress bar

        Returns:
            Dictionary containing:
                - frames: List of per-frame outputs
                - num_frames: Number of frames processed
                - selection: Person-selection metadata
        """
        import cv2

        all_outputs = []

        iterator = tqdm(frame_paths, desc="Processing frames") if progress else frame_paths

        for idx, frame_path in enumerate(iterator):
            # Load image (BGR to RGB)
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Warning: Could not read {frame_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process frame
            outputs = []
            selected_output = None
            selected_output_index = None
            inference_mode = "full"

            if self.single_person:
                if not self.selected_person_initialized:
                    outputs = self.process_frame(image, frame_idx=idx)
                    inference_mode = "select"
                    (
                        selected_output,
                        selected_output_index,
                        next_bbox,
                    ) = self._resolve_selected_output_from_full_detection(
                        image=image,
                        outputs=outputs,
                        frame_idx=idx,
                    )
                    if next_bbox is not None:
                        self.selected_bbox_prev = next_bbox.copy()
                elif self._should_run_full_refresh(idx):
                    outputs = self.process_frame(image, frame_idx=idx)
                    inference_mode = "refresh"
                    self.tracking_refresh_frames += 1
                    if self.tracking_consecutive_failures > 0:
                        self.tracking_recovery_events += 1

                    (
                        selected_output,
                        selected_output_index,
                        next_bbox,
                    ) = self._resolve_selected_output_from_full_detection(
                        image=image,
                        outputs=outputs,
                        frame_idx=idx,
                    )
                    if selected_output is not None and next_bbox is not None:
                        self.selected_bbox_prev = next_bbox.copy()
                        self.last_full_detection_frame = idx
                        self.tracking_consecutive_failures = 0
                else:
                    outputs = self.process_frame(
                        image,
                        frame_idx=idx,
                        bboxes=self.selected_bbox_prev,
                    )
                    inference_mode = "track"

                    tracked_output = outputs[0] if outputs else None
                    next_bbox = None
                    if tracked_output is not None:
                        next_bbox = self._next_tracking_bbox_from_tracked_output(
                            tracked_output,
                            image.shape,
                        )

                    if tracked_output is not None and next_bbox is not None:
                        selected_output = tracked_output
                        selected_output_index = self.selected_person_index_first_frame
                        self.selected_bbox_prev = next_bbox.copy()
                        self.tracking_consecutive_failures = 0
                        self.tracking_fast_path_frames += 1
                    else:
                        outputs = self.process_frame(image, frame_idx=idx)
                        inference_mode = "recovery"
                        self.tracking_refresh_frames += 1
                        self.tracking_recovery_events += 1

                        (
                            selected_output,
                            selected_output_index,
                            next_bbox,
                        ) = self._resolve_selected_output_from_full_detection(
                            image=image,
                            outputs=outputs,
                            frame_idx=idx,
                        )
                        if selected_output is not None and next_bbox is not None:
                            self.selected_bbox_prev = next_bbox.copy()
                            self.last_full_detection_frame = idx
                            self.tracking_consecutive_failures = 0
                        else:
                            self.tracking_consecutive_failures += 1
            else:
                outputs = self.process_frame(image, frame_idx=idx)
                if outputs:
                    selected_output = outputs[0]
                    selected_output_index = 0

            if progress and hasattr(iterator, "set_postfix_str"):
                if self.single_person:
                    if inference_mode == "track":
                        tracked_identity = self.selected_person_index_first_frame
                        postfix = f"mode=track, tracked={tracked_identity}"
                    else:
                        postfix = f"mode={inference_mode}, people={len(outputs)}"
                        if selected_output_index is not None:
                            postfix = f"{postfix}, selected={selected_output_index}"
                else:
                    postfix = f"people={len(outputs)}"
                iterator.set_postfix_str(postfix, refresh=False)

            if self.single_person and selected_output is not None:
                self._ensure_scene_ground_model(image, outputs, idx)
                self._attach_scene_ground_to_output(selected_output)

            all_outputs.append(
                {
                    "frame_idx": idx,
                    "frame_path": frame_path,
                    "outputs": outputs,
                    "output": selected_output,
                    "selected_output_index": selected_output_index,
                    "people_count": len(outputs),
                    "inference_mode": inference_mode,
                }
            )

        return {
            "frames": all_outputs,
            "num_frames": len(all_outputs),
            "selection": self.get_selection_metadata(),
            "scene_ground": self.get_scene_ground_metadata(),
        }

    def extract_keypoints_3d(
        self, frame_outputs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 3D keypoints from frame outputs into a single array.

        Args:
            frame_outputs: List of frame outputs from process_video

        Returns:
            Tuple of:
                - keypoints_3d: (T, 70, 3) array of 3D keypoints
                - valid_frames: (T,) boolean array indicating valid detections
        """
        num_frames = len(frame_outputs)
        keypoints_3d = np.zeros((num_frames, 70, 3), dtype=np.float32)
        valid_frames = np.zeros(num_frames, dtype=bool)

        for i, frame_data in enumerate(frame_outputs):
            if frame_data["output"] is not None:
                kp3d = frame_data["output"].get("pred_keypoints_3d")
                if kp3d is not None:
                    keypoints_3d[i] = np.array(kp3d)
                    valid_frames[i] = True

        return keypoints_3d, valid_frames

    def extract_mesh_vertices(
        self, frame_outputs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh vertices from frame outputs.

        Args:
            frame_outputs: List of frame outputs from process_video

        Returns:
            Tuple of:
                - vertices: (T, 6890, 3) array of mesh vertices
                - valid_frames: (T,) boolean array indicating valid detections
        """
        num_frames = len(frame_outputs)
        vertices = np.zeros((num_frames, 6890, 3), dtype=np.float32)
        valid_frames = np.zeros(num_frames, dtype=bool)

        for i, frame_data in enumerate(frame_outputs):
            if frame_data["output"] is not None:
                verts = frame_data["output"].get("pred_vertices")
                if verts is not None:
                    vertices[i] = np.array(verts)
                    valid_frames[i] = True

        return vertices, valid_frames

    def extract_camera_params(
        self, frame_outputs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Extract camera parameters from frame outputs.

        Args:
            frame_outputs: List of frame outputs from process_video

        Returns:
            Dictionary with:
                - focal_lengths: (T,) focal lengths
                - cam_translations: (T, 3) camera translations
        """
        num_frames = len(frame_outputs)
        focal_lengths = np.zeros(num_frames, dtype=np.float32)
        cam_translations = np.zeros((num_frames, 3), dtype=np.float32)

        for i, frame_data in enumerate(frame_outputs):
            if frame_data["output"] is not None:
                focal = frame_data["output"].get("focal_length", 1000.0)
                cam_t = frame_data["output"].get("pred_cam_t", [0, 0, 5])
                focal_lengths[i] = focal
                cam_translations[i] = np.array(cam_t)

        return {
            "focal_lengths": focal_lengths,
            "cam_translations": cam_translations,
        }
