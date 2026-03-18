"""
Post-processing utilities for keypoint sequences.

This module provides optional smoothing and bone normalization
for 3D keypoint sequences.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PostProcessor:
    """
    Post-processes 3D keypoint sequences.

    Provides:
    - Temporal smoothing (Butterworth filter)
    - Bone length normalization
    - Outlier detection and interpolation
    """

    # Standard bone connections for MHR70 skeleton
    # Format: (parent_idx, child_idx)
    BONE_CONNECTIONS = [
        # Spine
        (9, 10),  # Hip to hip (pelvis width, not a bone but a reference)
        (9, 11),  # Left hip to left knee
        (11, 13),  # Left knee to left ankle
        (10, 12),  # Right hip to right knee
        (12, 14),  # Right knee to right ankle
        # Upper body
        (69, 0),  # Neck to nose
        (67, 5),  # Left acromion to left shoulder
        (68, 6),  # Right acromion to right shoulder
        (5, 7),  # Left shoulder to left elbow
        (7, 62),  # Left elbow to left wrist
        (6, 8),  # Right shoulder to right elbow
        (8, 41),  # Right elbow to right wrist
        # Feet
        (13, 17),  # Left ankle to left heel
        (13, 15),  # Left ankle to left big toe
        (14, 20),  # Right ankle to right heel
        (14, 18),  # Right ankle to right big toe
    ]

    # Standard human proportions (relative to height)
    STANDARD_PROPORTIONS = {
        "upper_leg": 0.245,  # Thigh length / height
        "lower_leg": 0.246,  # Shank length / height
        "upper_arm": 0.186,  # Upper arm length / height
        "forearm": 0.146,  # Forearm length / height
        "shoulder_width": 0.259,  # Biacromial width / height
        "hip_width": 0.191,  # Bitrochanteric width / height
    }

    def __init__(
        self,
        smooth_filter: bool = False,
        filter_cutoff: float = 6.0,
        filter_order: int = 4,
        normalize_bones: bool = True,
    ):
        """
        Initialize post-processor.

        Args:
            smooth_filter: Whether to apply Butterworth filter
            filter_cutoff: Filter cutoff frequency in Hz
            filter_order: Filter order
            normalize_bones: Whether to normalize bone lengths
        """
        self.smooth_filter = smooth_filter
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.normalize_bones = normalize_bones

    def process(
        self,
        keypoints: np.ndarray,
        fps: float = 30.0,
        subject_height: float = 1.75,
    ) -> np.ndarray:
        """
        Apply post-processing to keypoint sequence.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            fps: Frame rate in Hz
            subject_height: Subject height in meters

        Returns:
            Processed keypoint sequence
        """
        processed = keypoints.copy()

        # Interpolate missing frames
        processed = self._interpolate_missing(processed)

        # Normalize bone lengths
        if self.normalize_bones:
            processed = self._normalize_bones(processed, subject_height)

        # Apply smoothing filter
        if self.smooth_filter:
            processed = self._apply_butterworth(processed, fps)

        return processed

    def _interpolate_missing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate frames with missing/invalid keypoints.

        Args:
            keypoints: (T, K, 3) keypoint sequence

        Returns:
            Interpolated sequence
        """
        result = keypoints.copy()
        T, K, _ = result.shape

        for k in range(K):
            for dim in range(3):
                values = result[:, k, dim]

                # Find valid (non-zero) values
                valid_mask = ~np.isnan(values) & (values != 0)

                if np.sum(valid_mask) < 2:
                    continue

                # Interpolate missing values
                valid_indices = np.where(valid_mask)[0]
                invalid_indices = np.where(~valid_mask)[0]

                if len(invalid_indices) > 0:
                    result[invalid_indices, k, dim] = np.interp(
                        invalid_indices, valid_indices, values[valid_indices]
                    )

        return result

    def _normalize_bones(
        self, keypoints: np.ndarray, subject_height: float
    ) -> np.ndarray:
        """
        Normalize bone lengths to anthropometric standards.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            subject_height: Subject height in meters

        Returns:
            Normalized sequence
        """
        result = keypoints.copy()

        # Calculate expected bone lengths based on subject height
        expected_lengths = {
            (9, 11): subject_height * self.STANDARD_PROPORTIONS["upper_leg"],
            (11, 13): subject_height * self.STANDARD_PROPORTIONS["lower_leg"],
            (10, 12): subject_height * self.STANDARD_PROPORTIONS["upper_leg"],
            (12, 14): subject_height * self.STANDARD_PROPORTIONS["lower_leg"],
            (5, 7): subject_height * self.STANDARD_PROPORTIONS["upper_arm"],
            (7, 62): subject_height * self.STANDARD_PROPORTIONS["forearm"],
            (6, 8): subject_height * self.STANDARD_PROPORTIONS["upper_arm"],
            (8, 41): subject_height * self.STANDARD_PROPORTIONS["forearm"],
        }

        # For each frame, scale bones to match expected lengths
        for t in range(result.shape[0]):
            for (parent_idx, child_idx), expected_len in expected_lengths.items():
                parent = result[t, parent_idx]
                child = result[t, child_idx]

                current_vec = child - parent
                current_len = np.linalg.norm(current_vec)

                if current_len > 0.01:  # Avoid division by zero
                    scale = expected_len / current_len
                    # Only scale if deviation is significant (>20%)
                    if abs(scale - 1.0) > 0.2:
                        scale = np.clip(scale, 0.8, 1.2)  # Limit correction
                        result[t, child_idx] = parent + current_vec * scale

        return result

    def _apply_butterworth(
        self, keypoints: np.ndarray, fps: float
    ) -> np.ndarray:
        """
        Apply Butterworth low-pass filter for smoothing.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            fps: Frame rate in Hz

        Returns:
            Smoothed sequence
        """
        from scipy.signal import butter, filtfilt

        # Design filter
        nyquist = fps / 2
        if self.filter_cutoff >= nyquist:
            # Cutoff too high, skip filtering
            return keypoints

        normalized_cutoff = self.filter_cutoff / nyquist
        b, a = butter(self.filter_order, normalized_cutoff, btype="low")

        result = keypoints.copy()
        T, K, _ = result.shape

        # Need at least 3x filter order samples
        min_samples = 3 * self.filter_order + 1
        if T < min_samples:
            return keypoints

        # Apply filter to each keypoint and dimension
        for k in range(K):
            for dim in range(3):
                try:
                    result[:, k, dim] = filtfilt(b, a, result[:, k, dim])
                except Exception:
                    # If filtering fails, keep original
                    pass

        return result

    def detect_outliers(
        self,
        keypoints: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        Detect outlier frames based on velocity.

        Args:
            keypoints: (T, K, 3) keypoint sequence
            threshold: Z-score threshold for outlier detection

        Returns:
            (T,) boolean mask where True indicates outlier
        """
        T = keypoints.shape[0]
        if T < 3:
            return np.zeros(T, dtype=bool)

        # Calculate frame-to-frame velocity
        velocities = np.diff(keypoints, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)

        # Average velocity per frame
        avg_velocity = np.mean(velocity_magnitudes, axis=1)

        # Z-score
        mean_vel = np.mean(avg_velocity)
        std_vel = np.std(avg_velocity)

        if std_vel < 1e-6:
            return np.zeros(T, dtype=bool)

        z_scores = np.abs(avg_velocity - mean_vel) / std_vel

        # Mark outliers (including the following frame)
        outliers = np.zeros(T, dtype=bool)
        outlier_indices = np.where(z_scores > threshold)[0]
        for idx in outlier_indices:
            outliers[idx] = True
            if idx + 1 < T:
                outliers[idx + 1] = True

        return outliers

    def fix_left_right_swaps(
        self, keypoints: np.ndarray
    ) -> np.ndarray:
        """
        Detect and fix left/right body part swapping.

        This can occur in monocular pose estimation when the
        model confuses left and right sides.

        Args:
            keypoints: (T, K, 3) keypoint sequence

        Returns:
            Corrected sequence
        """
        result = keypoints.copy()
        T = result.shape[0]

        # Pairs of left/right keypoints (MHR70 indices)
        lr_pairs = [
            (9, 10),  # hips
            (11, 12),  # knees
            (13, 14),  # ankles
            (5, 6),  # shoulders
            (7, 8),  # elbows
            (62, 41),  # wrists
        ]

        for t in range(1, T):
            swap_detected = False

            for left_idx, right_idx in lr_pairs:
                # Check if swapping would reduce velocity
                prev_left = result[t - 1, left_idx]
                prev_right = result[t - 1, right_idx]
                curr_left = result[t, left_idx]
                curr_right = result[t, right_idx]

                # Velocity with current assignment
                vel_normal = (
                    np.linalg.norm(curr_left - prev_left)
                    + np.linalg.norm(curr_right - prev_right)
                )

                # Velocity if swapped
                vel_swapped = (
                    np.linalg.norm(curr_right - prev_left)
                    + np.linalg.norm(curr_left - prev_right)
                )

                # If swapped velocity is significantly lower, there might be a swap
                if vel_swapped < vel_normal * 0.5:
                    swap_detected = True
                    break

            if swap_detected:
                # Swap all left/right pairs for this frame
                for left_idx, right_idx in lr_pairs:
                    result[t, [left_idx, right_idx]] = result[t, [right_idx, left_idx]]

        return result
