"""
Coordinate system transformation from camera to OpenSim.

This module handles the transformation of 3D coordinates from
SAM3D Body's camera-centric coordinate system to OpenSim's
biomechanical world coordinate system.
"""

from typing import Dict, Optional, Tuple
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
        }

    def get_last_ground_alignment_info(self) -> Dict[str, object]:
        """Return metadata about the most recent ground-alignment pass."""
        return dict(self.last_ground_alignment_info)

    def transform(
        self,
        keypoints_3d: np.ndarray,
        camera_translation: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        center_pelvis: bool = True,
        align_to_ground: bool = True,
        apply_global_translation: bool = False,
        ground_alignment_mode: str = "auto",
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

        Returns:
            Transformed keypoints in OpenSim coordinates
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        # Make a copy to avoid modifying input
        transformed = keypoints_3d.copy()

        # Apply rotation: Camera -> OpenSim axes
        for i in range(transformed.shape[0]):
            transformed[i] = transformed[i] @ self.CAMERA_TO_OPENSIM.T

        # Scale based on subject height (get scale factor for later use)
        transformed, height_scale = self._scale_to_subject(transformed, return_scale=True)

        # Apply global translation from cam_t
        if apply_global_translation and camera_translation is not None:
            transformed = self._apply_global_translation(
                transformed, camera_translation, height_scale
            )
        elif center_pelvis:
            # Center at pelvis (only if not using global translation)
            transformed = self._center_at_pelvis(transformed)

        # Align to ground
        if align_to_ground:
            transformed = self._align_to_ground(
                transformed,
                mode=ground_alignment_mode,
            )

        # Convert units
        transformed = transformed * self.scale_factor

        if single_frame:
            transformed = transformed[0]

        return transformed

    def _apply_global_translation(
        self,
        keypoints: np.ndarray,
        camera_translation: np.ndarray,
        height_scale: float,
    ) -> np.ndarray:
        """
        Apply global translation using cam_t from SAM3D Body.

        cam_t represents the camera translation vector which encodes the
        body's position in camera space. With moge2 FOV estimation, this
        should be accurate for tracking global movement.

        Args:
            keypoints: (N, K, 3) keypoints in OpenSim coordinates (pre-scaled)
            camera_translation: (N, 3) camera translations from SAM3D Body
                               Format: [x_right, y_down, z_forward] in camera space
            height_scale: Scale factor from height normalization

        Returns:
            Keypoints with global translation applied
        """
        num_frames = keypoints.shape[0]

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

        # Apply translation to keypoints
        for i in range(num_frames):
            # Relative translation from first frame
            delta_t = cam_t_smoothed[i] - first_frame_t
            # Apply X (forward) and Z (lateral) translation
            # Y (vertical) is handled by ground alignment
            keypoints[i, :, 0] += delta_t[0]  # X (forward)
            keypoints[i, :, 2] += delta_t[2]  # Z (lateral)

        return keypoints

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

    def _compute_contact_data(self, keypoints: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate stance/flight phases from heel and toe trajectories."""
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
        }

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
    ) -> np.ndarray:
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
            return self._align_to_ground_per_frame(aligned)

        vertical_offsets = np.zeros(num_frames, dtype=np.float32)
        first_contact_idx = int(np.flatnonzero(contact_mask)[0])
        current_offset = -float(min_foot_y[first_contact_idx])
        vertical_offsets[: first_contact_idx + 1] = current_offset

        for i in range(first_contact_idx, num_frames):
            if contact_mask[i]:
                current_offset = -float(min_foot_y[i])
            vertical_offsets[i] = current_offset

        for i in range(keypoints.shape[0]):
            aligned[i, :, 1] += vertical_offsets[i]

        return aligned

    def _align_to_ground(
        self,
        keypoints: np.ndarray,
        mode: str = "auto",
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

        contact_data = self._compute_contact_data(keypoints)
        longest_flight_run = self._longest_true_run(contact_data["flight"])

        applied_mode = requested_mode
        if requested_mode == "auto":
            applied_mode = "contact_aware" if longest_flight_run >= 3 else "per_frame_snap"

        if applied_mode == "contact_aware" and not np.any(contact_data["any_contact"]):
            applied_mode = "per_frame_snap"

        self.last_ground_alignment_info = {
            "requested_mode": requested_mode,
            "applied_mode": applied_mode,
            "contact_frames": int(np.sum(contact_data["any_contact"])),
            "flight_frames": int(np.sum(contact_data["flight"])),
            "longest_flight_run": int(longest_flight_run),
        }

        if applied_mode == "contact_aware":
            return self._align_to_ground_contact_aware(keypoints, contact_data)

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
