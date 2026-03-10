"""
Coordinate system transformation from camera to OpenSim.

This module handles the transformation of 3D coordinates from
SAM3D Body's camera-centric coordinate system to OpenSim's
biomechanical world coordinate system.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class CoordinateTransformer:
    """
    Transforms coordinates between SAM3D Body and OpenSim coordinate systems.

    SAM3D Body (Camera-centric):
        - X: Right
        - Y: Down
        - Z: Forward (depth into scene)

    OpenSim (Biomechanical world):
        - X: Forward (anterior)
        - Y: Up (superior)
        - Z: Right (lateral)
    """

    # Transformation matrix: Camera -> OpenSim
    # OpenSim X = Camera Z (forward)
    # OpenSim Y = -Camera Y (up = -down)
    # OpenSim Z = Camera X (right)
    CAMERA_TO_OPENSIM = np.array(
        [
            [0, 0, 1],  # X_opensim = Z_camera
            [0, -1, 0],  # Y_opensim = -Y_camera
            [1, 0, 0],  # Z_opensim = X_camera
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        subject_height: float = 1.75,
        units: str = "m",
    ):
        """
        Initialize coordinate transformer.

        Args:
            subject_height: Subject height in meters (for scaling)
            units: Output units ('m' for meters, 'mm' for millimeters)
        """
        self.subject_height = subject_height
        self.units = units
        self.scale_factor = 1000.0 if units == "mm" else 1.0
        self.last_ground_alignment_info: Dict[str, object] = {
            "requested_mode": None,
            "applied_mode": None,
            "contact_frames": 0,
            "flight_frames": 0,
            "longest_flight_run": 0,
            "raw_contact_frames": 0,
            "calibrated_contact_frames": 0,
            "scene_ground_used": False,
            "scene_ground_valid_frames": 0,
            "scene_ground_fused_frames": 0,
            "vertical_mode": "legacy_xz_only",
            "vertical_applied": False,
            "vertical_confident_frames": 0,
            "vertical_fallback_frames": 0,
            "max_cam_y_delta": 0.0,
            "max_clearance_floor": 0.0,
            "takeoff_events": 0,
            "landing_events": 0,
            "manual_plane_anchor_active": False,
            "manual_plane_left_bias_m": 0.0,
            "manual_plane_right_bias_m": 0.0,
            "manual_plane_calibration_frames": 0,
            "manual_plane_calibration_confidence": 0.0,
            "manual_plane_fallback_reason": None,
        }
        self.last_contact_data: Dict[str, object] = {}

    def get_last_ground_alignment_info(self) -> Dict[str, object]:
        """Return metadata about the most recent ground-alignment pass."""
        return dict(self.last_ground_alignment_info)

    def get_last_contact_data(self) -> Dict[str, object]:
        """Return the most recent per-frame contact metadata."""
        copied: Dict[str, object] = {}
        for key, value in self.last_contact_data.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def transform(
        self,
        keypoints_3d: np.ndarray,
        camera_translation: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        apply_global_translation: bool = False,
        ground_alignment_mode: str = "auto",
        scene_ground_data: Optional[Dict[str, Any]] = None,
        vertical_translation_mode: str = "auto",
    ) -> np.ndarray:
        """
        Transform keypoints from camera to OpenSim coordinates.

        Args:
            keypoints_3d: (N, K, 3) or (K, 3) array of 3D keypoints
            camera_translation: (N, 3) or (3,) camera translation vector(s)
                                Used when apply_global_translation=True
            focal_length: Camera focal length (for scaling)
            center_pelvis: Whether to center at pelvis (ignored if apply_global_translation=True)
            align_to_ground: Whether to align feet to ground plane
            apply_global_translation: Whether to use camera_translation for global movement
            ground_alignment_mode: Ground-alignment strategy:
                - 'auto': detect jump-like flight phases and preserve airborne Y motion
                - 'contact_aware': preserve Y during flight, only re-anchor during stance
                - 'per_frame_snap': legacy mode, lowest foot point is snapped to Y=0 every frame
            scene_ground_data: Optional MoGe-derived ground-plane / foot-clearance hints
            vertical_translation_mode: Vertical translation strategy:
                - 'auto': use hybrid Y when support-surface confidence is good
                - 'legacy_xz_only': keep current X/Z-only global translation
                - 'hybrid_support_plane': force hybrid Y when possible

        Returns:
            Transformed keypoints in OpenSim coordinates
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        # Make a copy to avoid modifying input
        transformed = keypoints_3d.copy()
        self.last_contact_data = {}

        # Apply rotation: Camera -> OpenSim axes
        for i in range(transformed.shape[0]):
            transformed[i] = transformed[i] @ self.CAMERA_TO_OPENSIM.T

        # Scale based on subject height (get scale factor for later use)
        transformed, height_scale = self._scale_to_subject(transformed, return_scale=True)

        translation_signals = None

        # Apply global translation from cam_t
        if apply_global_translation and camera_translation is not None:
            translation_signals = self._compute_global_translation_signals(
                camera_translation,
                height_scale,
                transformed.shape[0],
            )
            transformed = self._apply_horizontal_global_translation(
                transformed,
                translation_signals,
            )
        elif center_pelvis:
            # Center at pelvis (only if not using global translation)
            transformed = self._center_at_pelvis(transformed)

        # Align to ground
        if align_to_ground:
            transformed = self._align_to_ground(
                transformed,
                mode=ground_alignment_mode,
                scene_ground_data=scene_ground_data,
                translation_signals=translation_signals,
                vertical_translation_mode=vertical_translation_mode,
            )

        # Convert units
        transformed = transformed * self.scale_factor

        if single_frame:
            transformed = transformed[0]

        return transformed

    def _compute_global_translation_signals(
        self,
        camera_translation: np.ndarray,
        height_scale: float,
        num_frames: int,
    ) -> Dict[str, np.ndarray]:
        """
        Build smoothed cam_t translation signals in OpenSim coordinates.

        cam_t represents the camera translation vector which encodes the
        body's position in camera space. With moge2 FOV estimation, this
        should be accurate for tracking global movement.

        Args:
            camera_translation: (N, 3) camera translations from SAM3D Body
                               Format: [x_right, y_down, z_forward] in camera space
            height_scale: Scale factor from height normalization

        Returns:
            Dictionary containing smoothed camera translations and deltas
        """
        # Ensure camera_translation is 2D
        if camera_translation.ndim == 1:
            camera_translation = np.tile(camera_translation, (num_frames, 1))

        # Transform cam_t from camera space to OpenSim space
        # Camera: X=right, Y=down, Z=forward
        # OpenSim: X=forward, Y=up, Z=right
        cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T

        # Scale cam_t by the same height scale used for keypoints
        cam_t_opensim = cam_t_opensim * height_scale

        # Apply smoothing to reduce frame-to-frame jitter
        cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)

        # Compute relative translation (relative to first frame)
        # This centers the motion at the origin
        first_frame_t = cam_t_smoothed[0].copy()
        delta_t = cam_t_smoothed - first_frame_t[None, :]

        return {
            "cam_t_smoothed": cam_t_smoothed,
            "delta_t": delta_t,
            "delta_x": delta_t[:, 0],
            "delta_y": delta_t[:, 1],
            "delta_z": delta_t[:, 2],
        }

    def _apply_horizontal_global_translation(
        self,
        keypoints: np.ndarray,
        translation_signals: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply only horizontal global translation, preserving legacy behavior."""
        aligned = keypoints.copy()
        delta_x = np.asarray(translation_signals.get("delta_x"), dtype=np.float32)
        delta_z = np.asarray(translation_signals.get("delta_z"), dtype=np.float32)
        if delta_x.shape[0] != aligned.shape[0] or delta_z.shape[0] != aligned.shape[0]:
            return aligned

        aligned[:, :, 0] += delta_x[:, None]
        aligned[:, :, 2] += delta_z[:, None]
        return aligned

    def _smooth_cam_t(
        self,
        cam_t: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Apply smoothing to cam_t to reduce jitter.

        Args:
            cam_t: (N, 3) camera translations
            window_size: Smoothing window size

        Returns:
            Smoothed cam_t
        """
        from scipy.ndimage import uniform_filter1d

        smoothed = cam_t.copy()
        for axis in range(3):
            smoothed[:, axis] = uniform_filter1d(
                cam_t[:, axis], size=window_size, mode='nearest'
            )

        return smoothed

    def _detect_foot_contact(
        self,
        heel_positions: np.ndarray,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.02,
    ) -> np.ndarray:
        """
        Detect foot contact frames based on heel position and velocity.

        Args:
            heel_positions: (N, 3) heel positions over time
            height_threshold: Max height (m) to consider foot on ground
            velocity_threshold: Max velocity (m/frame) to consider foot stationary

        Returns:
            (N,) boolean array, True when foot is in contact
        """
        num_frames = len(heel_positions)
        contact = np.zeros(num_frames, dtype=bool)

        # Get vertical positions (Y axis in OpenSim)
        heights = heel_positions[:, 1]

        # Normalize heights relative to minimum (ground level)
        min_height = np.percentile(heights, 5)  # Use 5th percentile as ground
        rel_heights = heights - min_height

        # Compute velocity (frame-to-frame displacement)
        velocity = np.zeros(num_frames)
        for i in range(1, num_frames):
            displacement = np.linalg.norm(heel_positions[i] - heel_positions[i-1])
            velocity[i] = displacement

        # Smooth velocity with small window
        kernel_size = 3
        velocity_smooth = np.convolve(
            velocity, np.ones(kernel_size)/kernel_size, mode='same'
        )

        # Foot is in contact when:
        # 1. Height is low (near ground)
        # 2. Velocity is low (foot is stationary)
        for i in range(num_frames):
            height_ok = rel_heights[i] < height_threshold
            velocity_ok = velocity_smooth[i] < velocity_threshold
            contact[i] = height_ok and velocity_ok

        # Clean up contact signal - fill small gaps
        contact = self._clean_contact_signal(contact, min_gap=3, min_duration=5)

        return contact

    def _clean_contact_signal(
        self,
        contact: np.ndarray,
        min_gap: int = 3,
        min_duration: int = 5,
    ) -> np.ndarray:
        """
        Clean up contact detection signal by filling small gaps and
        removing short contacts.

        Args:
            contact: (N,) boolean contact signal
            min_gap: Fill gaps shorter than this
            min_duration: Remove contacts shorter than this

        Returns:
            Cleaned contact signal
        """
        contact = contact.copy()
        n = len(contact)

        # Fill small gaps (brief lift-offs during stance)
        i = 0
        while i < n:
            if not contact[i]:
                # Find end of gap
                j = i
                while j < n and not contact[j]:
                    j += 1
                gap_length = j - i
                # Fill if gap is small and surrounded by contact
                if gap_length <= min_gap and i > 0 and j < n:
                    contact[i:j] = True
                i = j
            else:
                i += 1

        # Remove short contacts (noise)
        i = 0
        while i < n:
            if contact[i]:
                # Find end of contact
                j = i
                while j < n and contact[j]:
                    j += 1
                duration = j - i
                # Remove if too short
                if duration < min_duration:
                    contact[i:j] = False
                i = j
            else:
                i += 1

        return contact

    @staticmethod
    def _longest_true_run(mask: np.ndarray) -> int:
        """Return the maximum length of consecutive True values."""
        longest = 0
        current = 0
        for is_true in mask.astype(bool):
            if is_true:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    @staticmethod
    def _is_manual_plane_anchor_enabled(
        scene_ground_data: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True when an accepted manual ROI plane should act as the stance anchor."""
        if not scene_ground_data or not scene_ground_data.get("available"):
            return False

        support_mode = str(scene_ground_data.get("support_surface_mode_applied", "")).lower()
        selection_status = str(
            scene_ground_data.get("support_surface_selection_status", "")
        ).lower()
        return support_mode == "manual_roi" and selection_status == "accepted"

    @staticmethod
    def _estimate_clearance_bias(
        clearance: np.ndarray,
        preferred_mask: np.ndarray,
        fallback_mask: np.ndarray,
        max_bias_m: float = 0.20,
    ) -> Tuple[float, int, str]:
        """
        Estimate a systematic foot-to-plane offset from stance-like frames.

        Preference order:
        1. pose-only contact candidates
        2. lower-tail of all valid clearances
        """
        preferred_values = np.asarray(clearance[preferred_mask], dtype=np.float32)
        preferred_values = preferred_values[np.isfinite(preferred_values)]
        if preferred_values.size >= 3:
            quantile = 30.0 if preferred_values.size >= 5 else 50.0
            bias = float(np.percentile(preferred_values, quantile))
            return float(np.clip(bias, 0.0, max_bias_m)), int(preferred_values.size), "pose-contact"

        fallback_values = np.asarray(clearance[fallback_mask], dtype=np.float32)
        fallback_values = fallback_values[np.isfinite(fallback_values)]
        if fallback_values.size == 0:
            return 0.0, 0, "unavailable"

        sorted_values = np.sort(fallback_values)
        tail_count = max(3, int(np.ceil(sorted_values.size * 0.03)))
        lower_tail = sorted_values[:tail_count]
        bias = float(np.median(lower_tail))
        return float(np.clip(bias, 0.0, max_bias_m)), int(fallback_values.size), "lower-tail"

    def _prepare_scene_ground_data_for_alignment(
        self,
        pose_contact_data: Dict[str, np.ndarray],
        scene_ground_data: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare scene-ground hints for alignment.

        For accepted manual ROI planes, calibrate a stance bias so the plane acts
        as the support anchor instead of a raw flight/contact sensor.
        """
        info: Dict[str, Any] = {
            "manual_plane_anchor_active": False,
            "manual_plane_left_bias_m": 0.0,
            "manual_plane_right_bias_m": 0.0,
            "manual_plane_calibration_frames": 0,
            "manual_plane_calibration_confidence": 0.0,
            "manual_plane_fallback_reason": None,
        }
        if not scene_ground_data:
            return scene_ground_data, info

        prepared: Dict[str, Any] = {}
        for key, value in scene_ground_data.items():
            if isinstance(value, np.ndarray):
                prepared[key] = value.copy()
            else:
                prepared[key] = value

        if not prepared.get("available"):
            return prepared, info

        left_clearance = np.asarray(
            prepared.get("left_clearance", np.array([], dtype=np.float32)),
            dtype=np.float32,
        )
        right_clearance = np.asarray(
            prepared.get("right_clearance", np.array([], dtype=np.float32)),
            dtype=np.float32,
        )
        valid_frames = np.asarray(
            prepared.get("valid_frames", np.zeros(len(left_clearance), dtype=bool)),
            dtype=bool,
        )
        if (
            left_clearance.shape != right_clearance.shape
            or left_clearance.shape[0] == 0
            or valid_frames.shape[0] != left_clearance.shape[0]
        ):
            info["manual_plane_fallback_reason"] = "invalid-scene-ground-shape"
            prepared.update(info)
            return prepared, info

        prepared["left_clearance_raw"] = left_clearance.copy()
        prepared["right_clearance_raw"] = right_clearance.copy()

        if not self._is_manual_plane_anchor_enabled(prepared):
            prepared.update(info)
            return prepared, info

        left_pose_contact = np.asarray(pose_contact_data["left_contact"], dtype=bool)
        right_pose_contact = np.asarray(pose_contact_data["right_contact"], dtype=bool)
        if (
            left_pose_contact.shape[0] != left_clearance.shape[0]
            or right_pose_contact.shape[0] != right_clearance.shape[0]
        ):
            info["manual_plane_fallback_reason"] = "pose-contact-shape-mismatch"
            prepared.update(info)
            return prepared, info

        plane_confidence = float(
            prepared.get(
                "clip_plane_confidence",
                np.nanmax(np.asarray(prepared.get("plane_confidence", [0.0]), dtype=np.float32)),
            )
        )
        valid_left = valid_frames & np.isfinite(left_clearance)
        valid_right = valid_frames & np.isfinite(right_clearance)

        left_bias, left_frames, left_source = self._estimate_clearance_bias(
            left_clearance,
            valid_left & left_pose_contact,
            valid_left,
        )
        right_bias, right_frames, right_source = self._estimate_clearance_bias(
            right_clearance,
            valid_right & right_pose_contact,
            valid_right,
        )
        calibration_frames = max(left_frames, right_frames)
        if calibration_frames <= 0:
            info["manual_plane_fallback_reason"] = "no-calibration-candidates"
            prepared.update(info)
            return prepared, info

        prepared["left_clearance"][valid_left] = np.maximum(
            left_clearance[valid_left] - left_bias,
            0.0,
        )
        prepared["right_clearance"][valid_right] = np.maximum(
            right_clearance[valid_right] - right_bias,
            0.0,
        )

        contact_threshold = float(prepared.get("contact_clearance_m", 0.04))
        flight_threshold = float(prepared.get("flight_clearance_m", 0.08))
        prepared["left_contact_hint"] = valid_left & (prepared["left_clearance"] <= contact_threshold)
        prepared["right_contact_hint"] = valid_right & (prepared["right_clearance"] <= contact_threshold)
        prepared["left_flight_hint"] = valid_left & (prepared["left_clearance"] >= flight_threshold)
        prepared["right_flight_hint"] = valid_right & (prepared["right_clearance"] >= flight_threshold)

        calibration_confidence = float(
            np.clip(plane_confidence, 0.0, 1.0)
            * min(1.0, calibration_frames / 8.0)
        )
        info.update(
            {
                "manual_plane_anchor_active": True,
                "manual_plane_left_bias_m": float(left_bias),
                "manual_plane_right_bias_m": float(right_bias),
                "manual_plane_calibration_frames": int(calibration_frames),
                "manual_plane_calibration_confidence": calibration_confidence,
                "manual_plane_fallback_reason": None,
            }
        )
        prepared.update(info)
        prepared["manual_plane_left_bias_source"] = left_source
        prepared["manual_plane_right_bias_source"] = right_source
        return prepared, info

    def _compute_contact_data_pose_only(
        self,
        keypoints: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Estimate stance/flight phases from heel and toe trajectories only."""
        left_heel_idx = 17
        right_heel_idx = 20
        left_toe_idx = 15
        right_toe_idx = 18

        left_heel = keypoints[:, left_heel_idx]
        right_heel = keypoints[:, right_heel_idx]
        left_toe = keypoints[:, left_toe_idx]
        right_toe = keypoints[:, right_toe_idx]

        left_contact = self._detect_foot_contact(
            left_heel,
            height_threshold=0.07,
            velocity_threshold=0.10,
        ) | self._detect_foot_contact(
            left_toe,
            height_threshold=0.07,
            velocity_threshold=0.10,
        )
        right_contact = self._detect_foot_contact(
            right_heel,
            height_threshold=0.07,
            velocity_threshold=0.10,
        ) | self._detect_foot_contact(
            right_toe,
            height_threshold=0.07,
            velocity_threshold=0.10,
        )

        left_contact = self._clean_contact_signal(left_contact, min_gap=2, min_duration=3)
        right_contact = self._clean_contact_signal(right_contact, min_gap=2, min_duration=3)
        any_contact = left_contact | right_contact
        flight = ~any_contact

        foot_indices = [left_heel_idx, right_heel_idx, left_toe_idx, right_toe_idx]
        min_foot_y = np.min(keypoints[:, foot_indices, 1], axis=1)

        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "any_contact": any_contact,
            "flight": flight,
            "min_foot_y": min_foot_y,
            "scene_ground_used": False,
            "scene_ground_valid_frames": 0,
            "scene_ground_fused_frames": 0,
        }

    def _fuse_scene_ground_contacts(
        self,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        scene_ground_data: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | bool]]:
        """Refine pose-only contact labels with MoGe-derived foot clearances."""
        if not scene_ground_data or not scene_ground_data.get("available"):
            return left_contact, right_contact, {
                "scene_ground_used": False,
                "scene_ground_valid_frames": 0,
                "scene_ground_fused_frames": 0,
            }

        valid_frames = np.asarray(
            scene_ground_data.get("valid_frames", np.zeros(len(left_contact), dtype=bool)),
            dtype=bool,
        )
        if valid_frames.shape[0] != left_contact.shape[0]:
            return left_contact, right_contact, {
                "scene_ground_used": False,
                "scene_ground_valid_frames": 0,
                "scene_ground_fused_frames": 0,
            }

        plane_confidence = np.asarray(
            scene_ground_data.get(
                "plane_confidence",
                np.zeros(len(left_contact), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        valid_frames &= plane_confidence >= 0.10
        if not np.any(valid_frames):
            return left_contact, right_contact, {
                "scene_ground_used": False,
                "scene_ground_valid_frames": 0,
                "scene_ground_fused_frames": 0,
            }

        pose_left_contact = np.asarray(left_contact, dtype=bool).copy()
        pose_right_contact = np.asarray(right_contact, dtype=bool).copy()
        pose_any_contact = pose_left_contact | pose_right_contact
        manual_plane_anchor_active = bool(
            scene_ground_data.get("manual_plane_anchor_active", False)
        )

        left_clearance = np.asarray(
            scene_ground_data.get("left_clearance", np.full(len(left_contact), np.nan)),
            dtype=np.float32,
        )
        right_clearance = np.asarray(
            scene_ground_data.get("right_clearance", np.full(len(right_contact), np.nan)),
            dtype=np.float32,
        )
        left_contact_hint = np.asarray(
            scene_ground_data.get(
                "left_contact_hint",
                np.zeros(len(left_contact), dtype=bool),
            ),
            dtype=bool,
        )
        right_contact_hint = np.asarray(
            scene_ground_data.get(
                "right_contact_hint",
                np.zeros(len(right_contact), dtype=bool),
            ),
            dtype=bool,
        )
        left_flight_hint = np.asarray(
            scene_ground_data.get(
                "left_flight_hint",
                np.zeros(len(left_contact), dtype=bool),
            ),
            dtype=bool,
        )
        right_flight_hint = np.asarray(
            scene_ground_data.get(
                "right_flight_hint",
                np.zeros(len(right_contact), dtype=bool),
            ),
            dtype=bool,
        )

        contact_threshold = float(scene_ground_data.get("contact_clearance_m", 0.04))
        flight_threshold = float(scene_ground_data.get("flight_clearance_m", 0.08))
        fused_left = left_contact.copy()
        fused_right = right_contact.copy()
        fused_frames = 0

        for idx in np.flatnonzero(valid_frames):
            left_valid = np.isfinite(left_clearance[idx])
            right_valid = np.isfinite(right_clearance[idx])
            if not (left_valid or right_valid):
                continue

            fused_frames += 1
            if manual_plane_anchor_active:
                if left_valid and (
                    left_contact_hint[idx] or left_clearance[idx] <= contact_threshold
                ):
                    fused_left[idx] = True
                if right_valid and (
                    right_contact_hint[idx] or right_clearance[idx] <= contact_threshold
                ):
                    fused_right[idx] = True

                strong_dual_flight = (
                    left_valid
                    and right_valid
                    and (left_flight_hint[idx] or left_clearance[idx] >= flight_threshold)
                    and (right_flight_hint[idx] or right_clearance[idx] >= flight_threshold)
                    and not pose_any_contact[idx]
                )
                if strong_dual_flight:
                    fused_left[idx] = False
                    fused_right[idx] = False
                continue

            if left_valid:
                if left_contact_hint[idx] or left_clearance[idx] <= contact_threshold:
                    fused_left[idx] = True
                elif left_flight_hint[idx] or left_clearance[idx] >= flight_threshold:
                    fused_left[idx] = False
            if right_valid:
                if right_contact_hint[idx] or right_clearance[idx] <= contact_threshold:
                    fused_right[idx] = True
                elif right_flight_hint[idx] or right_clearance[idx] >= flight_threshold:
                    fused_right[idx] = False

            if (
                left_valid
                and right_valid
                and left_clearance[idx] >= flight_threshold
                and right_clearance[idx] >= flight_threshold
            ):
                fused_left[idx] = False
                fused_right[idx] = False

        fused_left = self._clean_contact_signal(fused_left, min_gap=2, min_duration=3)
        fused_right = self._clean_contact_signal(fused_right, min_gap=2, min_duration=3)

        return fused_left, fused_right, {
            "scene_ground_used": fused_frames > 0,
            "scene_ground_valid_frames": int(np.sum(valid_frames)),
            "scene_ground_fused_frames": int(fused_frames),
        }

    def _compute_contact_data(
        self,
        keypoints: np.ndarray,
        scene_ground_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """Estimate stance/flight phases from pose plus optional scene-ground hints."""
        contact_data = self._compute_contact_data_pose_only(keypoints)
        prepared_scene_ground, manual_plane_info = self._prepare_scene_ground_data_for_alignment(
            contact_data,
            scene_ground_data,
        )
        left_contact, right_contact, scene_stats = self._fuse_scene_ground_contacts(
            contact_data["left_contact"],
            contact_data["right_contact"],
            prepared_scene_ground,
        )
        if manual_plane_info.get("manual_plane_anchor_active") and not np.any(left_contact | right_contact):
            if prepared_scene_ground is not None:
                valid_frames = np.asarray(
                    prepared_scene_ground.get(
                        "valid_frames", np.zeros(len(left_contact), dtype=bool)
                    ),
                    dtype=bool,
                )
                reconstructed_left = valid_frames & np.asarray(
                    prepared_scene_ground.get(
                        "left_contact_hint", np.zeros(len(left_contact), dtype=bool)
                    ),
                    dtype=bool,
                )
                reconstructed_right = valid_frames & np.asarray(
                    prepared_scene_ground.get(
                        "right_contact_hint", np.zeros(len(right_contact), dtype=bool)
                    ),
                    dtype=bool,
                )
                reconstructed_left = self._clean_contact_signal(
                    reconstructed_left, min_gap=2, min_duration=2
                )
                reconstructed_right = self._clean_contact_signal(
                    reconstructed_right, min_gap=2, min_duration=2
                )
                if np.any(reconstructed_left | reconstructed_right):
                    left_contact = reconstructed_left
                    right_contact = reconstructed_right
                    manual_plane_info["manual_plane_fallback_reason"] = "reconstructed-from-clearance"
                else:
                    manual_plane_info["manual_plane_fallback_reason"] = "no-stance-after-calibration"

        raw_contact_frames = int(np.sum(contact_data["any_contact"]))
        contact_data["left_contact"] = left_contact
        contact_data["right_contact"] = right_contact
        contact_data["any_contact"] = left_contact | right_contact
        contact_data["flight"] = ~contact_data["any_contact"]
        contact_data.update(scene_stats)
        contact_data["raw_contact_frames"] = raw_contact_frames
        contact_data["calibrated_contact_frames"] = int(np.sum(contact_data["any_contact"]))
        contact_data.update(manual_plane_info)
        return contact_data, prepared_scene_ground

    @staticmethod
    def _count_rising_edges(mask: np.ndarray) -> int:
        """Count False -> True transitions in a boolean signal."""
        signal = np.asarray(mask, dtype=bool)
        if signal.size == 0:
            return 0
        return int(np.sum(signal[1:] & ~signal[:-1])) + int(signal[0])

    @staticmethod
    def _count_falling_edges(mask: np.ndarray) -> int:
        """Count True -> False transitions in a boolean signal."""
        signal = np.asarray(mask, dtype=bool)
        if signal.size <= 1:
            return 0
        return int(np.sum(~signal[1:] & signal[:-1]))

    @staticmethod
    def _compute_contact_aware_vertical_offsets(
        contact_mask: np.ndarray,
        min_foot_y: np.ndarray,
    ) -> np.ndarray:
        """Compute the legacy contact-aware vertical offsets without mutating keypoints."""
        num_frames = len(min_foot_y)
        if num_frames == 0:
            return np.zeros(0, dtype=np.float32)

        vertical_offsets = np.zeros(num_frames, dtype=np.float32)
        if not np.any(contact_mask):
            return -np.asarray(min_foot_y, dtype=np.float32)

        first_contact_idx = int(np.flatnonzero(contact_mask)[0])
        current_offset = -float(min_foot_y[first_contact_idx])
        vertical_offsets[: first_contact_idx + 1] = current_offset

        for i in range(first_contact_idx, num_frames):
            if contact_mask[i]:
                current_offset = -float(min_foot_y[i])
            vertical_offsets[i] = current_offset

        return vertical_offsets

    def _project_scene_clearance_to_vertical(
        self,
        scene_ground_data: Optional[Dict[str, Any]],
        num_frames: int,
    ) -> Dict[str, Any]:
        """Project plane-normal foot clearances into approximate OpenSim Y clearance."""
        empty = {
            "available": False,
            "vertical_scale": 1.0,
            "clearance_floor": np.full(num_frames, np.nan, dtype=np.float32),
            "valid": np.zeros(num_frames, dtype=bool),
            "clip_confidence": 0.0,
        }
        if not scene_ground_data or num_frames <= 0:
            return empty

        left_clearance = np.asarray(
            scene_ground_data.get("left_clearance", np.full(num_frames, np.nan)),
            dtype=np.float32,
        )
        right_clearance = np.asarray(
            scene_ground_data.get("right_clearance", np.full(num_frames, np.nan)),
            dtype=np.float32,
        )
        valid_frames = np.asarray(
            scene_ground_data.get("valid_frames", np.zeros(num_frames, dtype=bool)),
            dtype=bool,
        )
        if left_clearance.shape[0] != num_frames or right_clearance.shape[0] != num_frames:
            return empty

        clip_confidence = float(
            scene_ground_data.get(
                "clip_plane_confidence",
                np.nanmax(np.asarray(scene_ground_data.get("plane_confidence", [0.0]), dtype=np.float32)),
            )
        )
        normal_cam = scene_ground_data.get("normal_cam")
        vertical_scale = 1.0
        if normal_cam is not None:
            normal_cam_arr = np.asarray(normal_cam, dtype=np.float32).reshape(-1)
            if normal_cam_arr.size == 3 and np.isfinite(normal_cam_arr).all():
                normal_opensim = normal_cam_arr @ self.CAMERA_TO_OPENSIM.T
                normal_norm = float(np.linalg.norm(normal_opensim))
                if normal_norm > 1e-6:
                    vertical_scale = abs(float(normal_opensim[1] / normal_norm))

        left_valid = valid_frames & np.isfinite(left_clearance)
        right_valid = valid_frames & np.isfinite(right_clearance)
        clearance_floor = np.full(num_frames, np.nan, dtype=np.float32)
        both_valid = left_valid & right_valid
        clearance_floor[both_valid] = np.minimum(
            left_clearance[both_valid],
            right_clearance[both_valid],
        )
        clearance_floor[left_valid & ~right_valid] = left_clearance[left_valid & ~right_valid]
        clearance_floor[right_valid & ~left_valid] = right_clearance[right_valid & ~left_valid]
        clearance_floor = clearance_floor * float(vertical_scale)
        valid = np.isfinite(clearance_floor)

        return {
            "available": bool(np.any(valid)),
            "vertical_scale": float(vertical_scale),
            "clearance_floor": clearance_floor,
            "valid": valid,
            "clip_confidence": clip_confidence,
        }

    def _compute_hybrid_vertical_offsets(
        self,
        contact_data: Dict[str, np.ndarray],
        translation_signals: Optional[Dict[str, np.ndarray]],
        scene_ground_data: Optional[Dict[str, Any]],
        vertical_translation_mode: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Blend cam_t_y and support-plane clearance into a vertical offset signal."""
        contact_mask = np.asarray(contact_data["any_contact"], dtype=bool)
        flight_mask = np.asarray(contact_data["flight"], dtype=bool)
        min_foot_y = np.asarray(contact_data["min_foot_y"], dtype=np.float32)
        num_frames = len(min_foot_y)
        base_offsets = self._compute_contact_aware_vertical_offsets(contact_mask, min_foot_y)

        info = {
            "vertical_mode": "legacy_xz_only",
            "vertical_applied": False,
            "vertical_confident_frames": 0,
            "vertical_fallback_frames": int(np.sum(flight_mask)),
            "max_cam_y_delta": 0.0,
            "max_clearance_floor": 0.0,
            "takeoff_events": self._count_rising_edges(flight_mask),
            "landing_events": self._count_falling_edges(flight_mask),
        }

        if vertical_translation_mode == "legacy_xz_only":
            return base_offsets, info
        if translation_signals is None:
            return base_offsets, info

        delta_y = np.asarray(translation_signals.get("delta_y", np.zeros(num_frames)), dtype=np.float32)
        if delta_y.shape[0] != num_frames:
            return base_offsets, info

        scene_vertical = self._project_scene_clearance_to_vertical(scene_ground_data, num_frames)
        confidence = scene_vertical.get("clip_confidence", 0.0) * scene_vertical.get("vertical_scale", 1.0)
        should_use_hybrid = (
            scene_vertical.get("available", False)
            and np.any(flight_mask)
            and scene_vertical.get("vertical_scale", 0.0) >= 0.35
            and confidence >= 0.12
        )
        if vertical_translation_mode == "hybrid_support_plane" and scene_vertical.get("available", False):
            should_use_hybrid = True
        if not should_use_hybrid:
            return base_offsets, info

        clearance_floor = np.asarray(scene_vertical["clearance_floor"], dtype=np.float32)
        clearance_valid = np.asarray(scene_vertical["valid"], dtype=bool)
        required_offset = np.full(num_frames, np.nan, dtype=np.float32)
        required_offset[clearance_valid] = clearance_floor[clearance_valid] - min_foot_y[clearance_valid]

        final_offsets = base_offsets.copy()
        blend_weight = float(np.clip(0.35 + 0.45 * confidence, 0.45, 0.80))
        transition_window = 2

        flight_indices = np.flatnonzero(flight_mask)
        if flight_indices.size == 0:
            return base_offsets, info

        segment_start = int(flight_indices[0])
        segment_prev = int(flight_indices[0])
        segments = []
        for idx in flight_indices[1:]:
            idx = int(idx)
            if idx == segment_prev + 1:
                segment_prev = idx
                continue
            segments.append((segment_start, segment_prev))
            segment_start = idx
            segment_prev = idx
        segments.append((segment_start, segment_prev))

        confident_frames = 0
        for start, end in segments:
            takeoff_idx = start - 1
            while takeoff_idx >= 0 and not contact_mask[takeoff_idx]:
                takeoff_idx -= 1
            if takeoff_idx < 0:
                takeoff_idx = start

            cam_reference = float(delta_y[takeoff_idx])
            base_takeoff_offset = float(base_offsets[takeoff_idx])
            for idx in range(start, end + 1):
                offset_cam = base_takeoff_offset + float(delta_y[idx] - cam_reference)
                if clearance_valid[idx]:
                    floor_offset = float(required_offset[idx])
                    blended = max(
                        floor_offset,
                        blend_weight * offset_cam + (1.0 - blend_weight) * floor_offset,
                    )
                else:
                    floor_offset = float("nan")
                    blended = offset_cam

                alpha = 1.0
                if idx - start < transition_window:
                    alpha = min(alpha, float(idx - start + 1) / float(transition_window + 1))
                if end - idx < transition_window:
                    alpha = min(alpha, float(end - idx + 1) / float(transition_window + 1))

                final_offsets[idx] = float((1.0 - alpha) * base_offsets[idx] + alpha * blended)
                confident_frames += 1

        info.update(
            {
                "vertical_mode": "hybrid_support_plane",
                "vertical_applied": confident_frames > 0,
                "vertical_confident_frames": int(confident_frames),
                "vertical_fallback_frames": int(max(0, np.sum(flight_mask) - confident_frames)),
                "max_cam_y_delta": float(np.nanmax(np.abs(delta_y))) if delta_y.size else 0.0,
                "max_clearance_floor": float(np.nanmax(clearance_floor[clearance_valid])) if np.any(clearance_valid) else 0.0,
            }
        )
        return final_offsets, info

    def _compute_foot_anchored_translation(
        self,
        keypoints: np.ndarray,
        left_heel: np.ndarray,
        right_heel: np.ndarray,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
    ) -> np.ndarray:
        """
        Compute global translation by anchoring to planted feet.

        When a foot is planted (in contact), it should stay in the same
        world position. We use this constraint to derive stable global motion.

        Args:
            keypoints: (N, K, 3) keypoints
            left_heel: (N, 3) left heel positions
            right_heel: (N, 3) right heel positions
            left_contact: (N,) left foot contact flags
            right_contact: (N,) right foot contact flags

        Returns:
            (N, 2) global XZ offset per frame
        """
        num_frames = len(keypoints)
        global_offset = np.zeros((num_frames, 2))  # X, Z offsets

        # Track world position of each foot when planted
        left_world_pos = np.array([0.0, 0.0])  # X, Z in world
        right_world_pos = np.array([0.0, 0.0])

        # Initialize with first frame positions
        left_world_pos[0] = left_heel[0, 0]  # X
        left_world_pos[1] = left_heel[0, 2]  # Z
        right_world_pos[0] = right_heel[0, 0]
        right_world_pos[1] = right_heel[0, 2]

        # Track which foot was last used as anchor
        last_anchor = None  # 'left', 'right', or None

        for i in range(num_frames):
            left_local = np.array([left_heel[i, 0], left_heel[i, 2]])
            right_local = np.array([right_heel[i, 0], right_heel[i, 2]])

            if left_contact[i] and right_contact[i]:
                # Both feet planted - use average
                left_offset = left_world_pos - left_local
                right_offset = right_world_pos - right_local
                global_offset[i] = (left_offset + right_offset) / 2
                last_anchor = 'both'

            elif left_contact[i]:
                # Left foot planted - anchor to it
                global_offset[i] = left_world_pos - left_local
                # Update right foot world position
                right_world_pos = right_local + global_offset[i]
                last_anchor = 'left'

            elif right_contact[i]:
                # Right foot planted - anchor to it
                global_offset[i] = right_world_pos - right_local
                # Update left foot world position
                left_world_pos = left_local + global_offset[i]
                last_anchor = 'right'

            else:
                # Neither foot planted (flight phase) - interpolate
                if i > 0:
                    # Use previous offset (maintain momentum)
                    global_offset[i] = global_offset[i-1]
                # Update both foot world positions
                left_world_pos = left_local + global_offset[i]
                right_world_pos = right_local + global_offset[i]

        # Smooth the global offset to reduce any remaining jitter
        global_offset = self._smooth_global_offset(global_offset)

        return global_offset

    def _smooth_global_offset(
        self,
        offset: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Apply light smoothing to global offset to reduce jitter.

        Args:
            offset: (N, 2) XZ offset per frame
            window_size: Smoothing window size

        Returns:
            Smoothed offset
        """
        from scipy.ndimage import uniform_filter1d

        smoothed = offset.copy()
        smoothed[:, 0] = uniform_filter1d(offset[:, 0], size=window_size, mode='nearest')
        smoothed[:, 1] = uniform_filter1d(offset[:, 1], size=window_size, mode='nearest')

        return smoothed

    def _scale_to_subject(
        self, keypoints: np.ndarray, return_scale: bool = False
    ) -> np.ndarray:
        """
        Scale keypoints to match subject height.

        Uses the distance from ankles to head as reference.

        Args:
            keypoints: (N, K, 3) keypoints
            return_scale: Whether to return the scale factor used

        Returns:
            Scaled keypoints, or (scaled keypoints, scale factor) if return_scale=True
        """
        # Estimate current height from skeleton
        # Use average of head-to-ankle distances

        # Indices in MHR70: 0=nose, 13=left_ankle, 14=right_ankle
        head_idx = 0
        left_ankle_idx = 13
        right_ankle_idx = 14

        # Calculate height for each frame
        heights = []
        for i in range(keypoints.shape[0]):
            head = keypoints[i, head_idx]
            left_ankle = keypoints[i, left_ankle_idx]
            right_ankle = keypoints[i, right_ankle_idx]
            ankle_mid = (left_ankle + right_ankle) / 2
            height = np.linalg.norm(head - ankle_mid)
            if height > 0.1:  # Sanity check
                heights.append(height)

        scale = 1.0
        if heights:
            avg_height = np.mean(heights)
            # Add ~10% for feet-to-ankle and top-of-head
            estimated_full_height = avg_height * 1.1
            scale = self.subject_height / estimated_full_height
            keypoints = keypoints * scale

        if return_scale:
            return keypoints, scale
        return keypoints

    def _center_at_pelvis(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Center keypoints at pelvis (midpoint of hips).

        Args:
            keypoints: (N, K, 3) keypoints

        Returns:
            Centered keypoints
        """
        # Indices: 9=left_hip, 10=right_hip
        left_hip_idx = 9
        right_hip_idx = 10

        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, left_hip_idx] + keypoints[i, right_hip_idx]) / 2
            # Center X and Z, but keep Y relative to ground
            keypoints[i, :, 0] -= pelvis[0]
            keypoints[i, :, 2] -= pelvis[2]

        return keypoints

    def _align_to_ground_per_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Legacy alignment: lowest foot point is snapped to Y=0 every frame.

        This removes airborne height and is kept only as a compatibility mode.

        Args:
            keypoints: (N, K, 3) keypoints

        Returns:
            Ground-aligned keypoints
        """
        # Foot indices in MHR70
        left_heel_idx = 17
        right_heel_idx = 20
        left_toe_idx = 15   # left_big_toe
        right_toe_idx = 18  # right_big_toe

        aligned = keypoints.copy()

        # Per-frame alignment: lowest foot point at Y=0
        for i in range(aligned.shape[0]):
            foot_heights = [
                aligned[i, left_heel_idx, 1],
                aligned[i, right_heel_idx, 1],
                aligned[i, left_toe_idx, 1],
                aligned[i, right_toe_idx, 1],
            ]
            min_y = min(foot_heights)
            aligned[i, :, 1] -= min_y

        return aligned

    def _align_to_ground_contact_aware(
        self,
        keypoints: np.ndarray,
        contact_data: Dict[str, np.ndarray],
        translation_signals: Optional[Dict[str, np.ndarray]] = None,
        scene_ground_data: Optional[Dict[str, Any]] = None,
        vertical_translation_mode: str = "auto",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preserve jump height by only re-anchoring Y during stance frames.

        During flight, the most recent stance offset is held instead of snapping
        the current frame back to the ground plane.
        """
        aligned = keypoints.copy()
        contact_mask = contact_data["any_contact"]
        min_foot_y = contact_data["min_foot_y"]
        num_frames = len(aligned)

        if not np.any(contact_mask):
            return self._align_to_ground_per_frame(aligned), {
                "vertical_mode": "legacy_xz_only",
                "vertical_applied": False,
                "vertical_confident_frames": 0,
                "vertical_fallback_frames": int(np.sum(contact_data["flight"])),
                "max_cam_y_delta": 0.0,
                "max_clearance_floor": 0.0,
                "takeoff_events": self._count_rising_edges(contact_data["flight"]),
                "landing_events": self._count_falling_edges(contact_data["flight"]),
            }

        resolved_vertical_mode = (vertical_translation_mode or "auto").lower()
        if resolved_vertical_mode not in {"auto", "legacy_xz_only", "hybrid_support_plane"}:
            raise ValueError(
                f"Unsupported vertical translation mode: {vertical_translation_mode}. "
                "Expected one of ['auto', 'hybrid_support_plane', 'legacy_xz_only']."
            )
        if resolved_vertical_mode == "auto":
            resolved_vertical_mode = (
                "hybrid_support_plane"
                if translation_signals is not None and scene_ground_data and scene_ground_data.get("available")
                else "legacy_xz_only"
            )

        vertical_offsets, vertical_info = self._compute_hybrid_vertical_offsets(
            contact_data=contact_data,
            translation_signals=translation_signals,
            scene_ground_data=scene_ground_data,
            vertical_translation_mode=resolved_vertical_mode,
        )

        if vertical_offsets.shape[0] != num_frames:
            vertical_offsets = self._compute_contact_aware_vertical_offsets(contact_mask, min_foot_y)
            vertical_info = {
                "vertical_mode": "legacy_xz_only",
                "vertical_applied": False,
                "vertical_confident_frames": 0,
                "vertical_fallback_frames": int(np.sum(contact_data["flight"])),
                "max_cam_y_delta": 0.0,
                "max_clearance_floor": 0.0,
                "takeoff_events": self._count_rising_edges(contact_data["flight"]),
                "landing_events": self._count_falling_edges(contact_data["flight"]),
            }

        for i in range(num_frames):
            aligned[i, :, 1] += vertical_offsets[i]

        return aligned, vertical_info

    def _align_to_ground(
        self,
        keypoints: np.ndarray,
        mode: str = "auto",
        scene_ground_data: Optional[Dict[str, Any]] = None,
        translation_signals: Optional[Dict[str, np.ndarray]] = None,
        vertical_translation_mode: str = "auto",
    ) -> np.ndarray:
        """
        Align skeleton to ground using the requested strategy.

        Modes:
            per_frame_snap: legacy per-frame foot snap
            contact_aware: preserve airborne Y, re-anchor only during stance
            auto: choose contact_aware when a sustained flight phase is detected
        """
        requested_mode = (mode or "auto").lower()
        valid_modes = {"auto", "contact_aware", "per_frame_snap"}
        if requested_mode not in valid_modes:
            raise ValueError(
                f"Unsupported ground alignment mode: {mode}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        contact_data, prepared_scene_ground_data = self._compute_contact_data(
            keypoints,
            scene_ground_data=scene_ground_data,
        )
        self.last_contact_data = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in contact_data.items()
        }
        longest_flight_run = self._longest_true_run(contact_data["flight"])

        applied_mode = requested_mode
        if requested_mode == "auto":
            applied_mode = "contact_aware" if longest_flight_run >= 3 else "per_frame_snap"

        if applied_mode == "contact_aware" and not np.any(contact_data["any_contact"]):
            applied_mode = "per_frame_snap"

        vertical_info = {
            "vertical_mode": "legacy_xz_only",
            "vertical_applied": False,
            "vertical_confident_frames": 0,
            "vertical_fallback_frames": int(np.sum(contact_data["flight"])),
            "max_cam_y_delta": 0.0,
            "max_clearance_floor": 0.0,
            "takeoff_events": self._count_rising_edges(contact_data["flight"]),
            "landing_events": self._count_falling_edges(contact_data["flight"]),
        }

        self.last_ground_alignment_info = {
            "requested_mode": requested_mode,
            "applied_mode": applied_mode,
            "contact_frames": int(np.sum(contact_data["any_contact"])),
            "flight_frames": int(np.sum(contact_data["flight"])),
            "longest_flight_run": int(longest_flight_run),
            "raw_contact_frames": int(contact_data.get("raw_contact_frames", 0)),
            "calibrated_contact_frames": int(contact_data.get("calibrated_contact_frames", 0)),
            "scene_ground_used": bool(contact_data.get("scene_ground_used", False)),
            "scene_ground_valid_frames": int(contact_data.get("scene_ground_valid_frames", 0)),
            "scene_ground_fused_frames": int(contact_data.get("scene_ground_fused_frames", 0)),
            "manual_plane_anchor_active": bool(
                contact_data.get("manual_plane_anchor_active", False)
            ),
            "manual_plane_left_bias_m": float(
                contact_data.get("manual_plane_left_bias_m", 0.0)
            ),
            "manual_plane_right_bias_m": float(
                contact_data.get("manual_plane_right_bias_m", 0.0)
            ),
            "manual_plane_calibration_frames": int(
                contact_data.get("manual_plane_calibration_frames", 0)
            ),
            "manual_plane_calibration_confidence": float(
                contact_data.get("manual_plane_calibration_confidence", 0.0)
            ),
            "manual_plane_fallback_reason": contact_data.get(
                "manual_plane_fallback_reason"
            ),
            **vertical_info,
        }

        if applied_mode == "contact_aware":
            aligned, vertical_info = self._align_to_ground_contact_aware(
                keypoints,
                contact_data,
                translation_signals=translation_signals,
                scene_ground_data=prepared_scene_ground_data,
                vertical_translation_mode=vertical_translation_mode,
            )
            self.last_ground_alignment_info.update(vertical_info)
            return aligned

        return self._align_to_ground_per_frame(keypoints)

    def correct_forward_lean(
        self,
        keypoints: np.ndarray,
        angle: Optional[float] = None,
    ) -> np.ndarray:
        """
        Correct systematic forward/backward lean in pose estimates.

        Args:
            keypoints: (N, K, 3) keypoints in OpenSim coordinates
            angle: Lean angle in degrees (positive = forward lean)
                   If None, automatically estimates from pose.

        Returns:
            Corrected keypoints
        """
        single_frame = keypoints.ndim == 2
        if single_frame:
            keypoints = keypoints[np.newaxis, ...]

        if angle is None:
            angle = self._estimate_lean_angle(keypoints)

        if abs(angle) < 1.0:  # Less than 1 degree, no correction needed
            if single_frame:
                return keypoints[0]
            return keypoints

        # Create rotation matrix around Z axis (lateral axis in OpenSim)
        rad = np.radians(-angle)  # Negative to correct the lean
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ]
        )

        # Apply rotation around pelvis center
        left_hip_idx = 9
        right_hip_idx = 10

        corrected = keypoints.copy()
        for i in range(corrected.shape[0]):
            pelvis = (corrected[i, left_hip_idx] + corrected[i, right_hip_idx]) / 2
            # Translate to origin, rotate, translate back
            corrected[i] = corrected[i] - pelvis
            corrected[i] = corrected[i] @ rotation.T
            corrected[i] = corrected[i] + pelvis

        if single_frame:
            corrected = corrected[0]

        return corrected

    def _estimate_lean_angle(self, keypoints: np.ndarray) -> float:
        """
        Estimate forward lean angle from pose.

        Uses the angle between pelvis-thorax line and vertical.

        Args:
            keypoints: (N, K, 3) keypoints

        Returns:
            Estimated lean angle in degrees
        """
        # Indices: 9/10=hips, 67/68=acromions
        left_hip_idx = 9
        right_hip_idx = 10
        left_acromion_idx = 67
        right_acromion_idx = 68

        angles = []
        for i in range(keypoints.shape[0]):
            pelvis = (keypoints[i, left_hip_idx] + keypoints[i, right_hip_idx]) / 2
            thorax = (
                keypoints[i, left_acromion_idx] + keypoints[i, right_acromion_idx]
            ) / 2

            # Vector from pelvis to thorax
            spine_vec = thorax - pelvis
            spine_vec_xz = np.array([spine_vec[0], spine_vec[1]])  # X (forward) and Y (up)

            # Angle with vertical (Y axis)
            if np.linalg.norm(spine_vec_xz) > 0.01:
                vertical = np.array([0, 1])
                cos_angle = np.dot(spine_vec_xz, vertical) / (
                    np.linalg.norm(spine_vec_xz) * np.linalg.norm(vertical)
                )
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                # Determine sign (forward vs backward lean)
                if spine_vec[0] > 0:  # Forward lean
                    angles.append(angle)
                else:
                    angles.append(-angle)

        if angles:
            return np.median(angles)
        return 0.0

    def batch_transform(
        self,
        keypoints_sequence: np.ndarray,
        camera_translations: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Transform a sequence of keypoints.

        Args:
            keypoints_sequence: (T, K, 3) sequence of keypoints
            camera_translations: (T, 3) camera translations per frame
            **kwargs: Additional arguments passed to transform()

        Returns:
            (T, K, 3) transformed sequence
        """
        return self.transform(
            keypoints_sequence,
            camera_translation=camera_translations,
            **kwargs
        )
