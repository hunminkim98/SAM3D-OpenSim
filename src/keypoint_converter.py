"""
MHR70 to OpenSim marker conversion.

This module handles the mapping from SAM3D Body's MHR70 keypoints
to OpenSim-compatible marker positions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml


class KeypointConverter:
    """
    Converts MHR70 keypoints to OpenSim markers.

    Handles:
    - Direct keypoint mapping
    - Derived marker computation (midpoints, interpolations)
    - Marker naming for OpenSim compatibility
    """

    # MHR70 keypoint names in order (indices 0-69)
    MHR70_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
        # Right hand (21 points, indices 21-41)
        "right_thumb_tip",
        "right_thumb_first_joint",
        "right_thumb_second_joint",
        "right_thumb_third_joint",
        "right_index_tip",
        "right_index_first_joint",
        "right_index_second_joint",
        "right_index_third_joint",
        "right_middle_tip",
        "right_middle_first_joint",
        "right_middle_second_joint",
        "right_middle_third_joint",
        "right_ring_tip",
        "right_ring_first_joint",
        "right_ring_second_joint",
        "right_ring_third_joint",
        "right_pinky_tip",
        "right_pinky_first_joint",
        "right_pinky_second_joint",
        "right_pinky_third_joint",
        "right_wrist",
        # Left hand (21 points, indices 42-62)
        "left_thumb_tip",
        "left_thumb_first_joint",
        "left_thumb_second_joint",
        "left_thumb_third_joint",
        "left_index_tip",
        "left_index_first_joint",
        "left_index_second_joint",
        "left_index_third_joint",
        "left_middle_tip",
        "left_middle_first_joint",
        "left_middle_second_joint",
        "left_middle_third_joint",
        "left_ring_tip",
        "left_ring_first_joint",
        "left_ring_second_joint",
        "left_ring_third_joint",
        "left_pinky_tip",
        "left_pinky_first_joint",
        "left_pinky_second_joint",
        "left_pinky_third_joint",
        "left_wrist",
        # Extra anatomical (7 points, indices 63-69)
        "left_olecranon",
        "right_olecranon",
        "left_cubital_fossa",
        "right_cubital_fossa",
        "left_acromion",
        "right_acromion",
        "neck",
    ]

    # Default MHR70 to OpenSim marker mapping
    DEFAULT_OPENSIM_NAMES = {
        # Head
        0: "Nose",
        1: "LEye",
        2: "REye",
        3: "LEar",
        4: "REar",
        # Torso
        5: "LShoulder",
        6: "RShoulder",
        69: "Neck",
        67: "LAcromion",
        68: "RAcromion",
        # Arms
        7: "LElbow",
        8: "RElbow",
        63: "LOlecranon",
        64: "ROlecranon",
        65: "LCubitalFossa",
        66: "RCubitalFossa",
        62: "LWrist",
        41: "RWrist",
        # Pelvis
        9: "LHip",
        10: "RHip",
        # Legs
        11: "LKnee",
        12: "RKnee",
        13: "LAnkle",
        14: "RAnkle",
        # Feet
        15: "LBigToe",
        16: "LSmallToe",
        17: "LHeel",
        18: "RBigToe",
        19: "RSmallToe",
        20: "RHeel",
        # Right hand
        21: "RThumbTip",
        22: "RThumb1",
        23: "RThumb2",
        24: "RThumb3",
        25: "RIndexTip",
        26: "RIndex1",
        27: "RIndex2",
        28: "RIndex3",
        29: "RMiddleTip",
        30: "RMiddle1",
        31: "RMiddle2",
        32: "RMiddle3",
        33: "RRingTip",
        34: "RRing1",
        35: "RRing2",
        36: "RRing3",
        37: "RPinkyTip",
        38: "RPinky1",
        39: "RPinky2",
        40: "RPinky3",
        # Left hand
        42: "LThumbTip",
        43: "LThumb1",
        44: "LThumb2",
        45: "LThumb3",
        46: "LIndexTip",
        47: "LIndex1",
        48: "LIndex2",
        49: "LIndex3",
        50: "LMiddleTip",
        51: "LMiddle1",
        52: "LMiddle2",
        53: "LMiddle3",
        54: "LRingTip",
        55: "LRing1",
        56: "LRing2",
        57: "LRing3",
        58: "LPinkyTip",
        59: "LPinky1",
        60: "LPinky2",
        61: "LPinky3",
    }

    def __init__(self, mapping_path: Optional[str] = None):
        """
        Initialize keypoint converter.

        Args:
            mapping_path: Path to custom marker mapping YAML file
        """
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, "r") as f:
                mapping_config = yaml.safe_load(f)
            self.opensim_names = {
                int(k): v for k, v in mapping_config.get("opensim_markers", {}).items()
            }
            self.derived_markers = mapping_config.get("derived_markers", {})
            self.marker_weights = mapping_config.get("marker_weights", {})
        else:
            self.opensim_names = self.DEFAULT_OPENSIM_NAMES.copy()
            self.derived_markers = self._get_default_derived_markers()
            self.marker_weights = {}

    def _get_default_derived_markers(self) -> dict:
        """Get default derived marker definitions."""
        return {
            "PelvisCenter": {
                "type": "midpoint",
                "points": [9, 10],  # left_hip, right_hip
            },
            "Thorax": {
                "type": "midpoint",
                "points": [67, 68],  # left_acromion, right_acromion
            },
            "SpineMid": {
                "type": "interpolate",
                "points": [9, 10, 67, 68],
                "ratio": 0.5,
            },
        }

    def convert(
        self,
        keypoints_3d: np.ndarray,
        include_derived: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert MHR70 keypoints to OpenSim markers.

        Args:
            keypoints_3d: (N, 70, 3) or (70, 3) array of MHR70 keypoints
            include_derived: Whether to include derived markers

        Returns:
            Tuple of:
                - markers: (N, M, 3) or (M, 3) array of OpenSim markers
                - marker_names: List of marker names in order
        """
        single_frame = keypoints_3d.ndim == 2
        if single_frame:
            keypoints_3d = keypoints_3d[np.newaxis, ...]

        num_frames = keypoints_3d.shape[0]

        # Get direct markers
        marker_list = []
        marker_names = []

        for mhr_idx in sorted(self.opensim_names.keys()):
            marker_list.append(keypoints_3d[:, mhr_idx, :])
            marker_names.append(self.opensim_names[mhr_idx])

        # Add derived markers
        if include_derived:
            for name, definition in self.derived_markers.items():
                derived = self._compute_derived_marker(keypoints_3d, definition)
                marker_list.append(derived)
                marker_names.append(name)

        markers = np.stack(marker_list, axis=1)

        if single_frame:
            markers = markers[0]

        return markers, marker_names

    def _compute_derived_marker(
        self, keypoints_3d: np.ndarray, definition: dict
    ) -> np.ndarray:
        """
        Compute a derived marker from keypoints.

        Args:
            keypoints_3d: (N, 70, 3) keypoints
            definition: Marker definition dict with 'type' and 'points'

        Returns:
            (N, 3) computed marker positions
        """
        marker_type = definition["type"]
        points = definition["points"]

        if marker_type == "midpoint":
            # Midpoint of two keypoints
            p1 = keypoints_3d[:, points[0], :]
            p2 = keypoints_3d[:, points[1], :]
            return (p1 + p2) / 2

        elif marker_type == "interpolate":
            # Interpolate between two groups
            ratio = definition.get("ratio", 0.5)
            if len(points) == 4:
                # Midpoint of first two, midpoint of last two, then interpolate
                p1 = (keypoints_3d[:, points[0], :] + keypoints_3d[:, points[1], :]) / 2
                p2 = (keypoints_3d[:, points[2], :] + keypoints_3d[:, points[3], :]) / 2
                return p1 * (1 - ratio) + p2 * ratio
            else:
                p1 = keypoints_3d[:, points[0], :]
                p2 = keypoints_3d[:, points[1], :]
                return p1 * (1 - ratio) + p2 * ratio

        elif marker_type == "offset":
            # Offset from a keypoint
            base = keypoints_3d[:, points[0], :]
            offset = np.array(definition.get("offset", [0, 0, 0]))
            return base + offset

        else:
            raise ValueError(f"Unknown derived marker type: {marker_type}")

    def get_marker_names(self, include_derived: bool = True) -> List[str]:
        """
        Get list of marker names in order.

        Args:
            include_derived: Whether to include derived markers

        Returns:
            List of marker names
        """
        names = [self.opensim_names[idx] for idx in sorted(self.opensim_names.keys())]
        if include_derived:
            names.extend(self.derived_markers.keys())
        return names

    def get_marker_weights(self) -> Dict[str, float]:
        """
        Get IK marker weights.

        Returns:
            Dictionary mapping marker names to weights (0-1)
        """
        # Default weights
        default_weights = {
            # High weight - reliable
            "LHip": 1.0,
            "RHip": 1.0,
            "LShoulder": 1.0,
            "RShoulder": 1.0,
            "LKnee": 1.0,
            "RKnee": 1.0,
            "LAnkle": 1.0,
            "RAnkle": 1.0,
            # Medium weight
            "LElbow": 0.8,
            "RElbow": 0.8,
            "LWrist": 0.8,
            "RWrist": 0.8,
            "Neck": 0.8,
            # Lower weight - less stable
            "Nose": 0.5,
            "LEye": 0.3,
            "REye": 0.3,
            "LEar": 0.3,
            "REar": 0.3,
            # Feet
            "LBigToe": 0.5,
            "LSmallToe": 0.35,
            "RBigToe": 0.5,
            "RSmallToe": 0.35,
            "LHeel": 0.6,
            "RHeel": 0.6,
            # Derived
            "PelvisCenter": 1.0,
            "Thorax": 0.8,
        }

        # Override with custom weights
        weights = default_weights.copy()
        weights.update(self.marker_weights)

        # Set default for finger markers
        for name in self.get_marker_names():
            if name not in weights:
                if "Thumb" in name or "Index" in name or "Middle" in name or "Ring" in name or "Pinky" in name:
                    weights[name] = 0.3
                else:
                    weights[name] = 0.5

        return weights

    @staticmethod
    def get_mhr70_name(index: int) -> str:
        """Get MHR70 keypoint name by index."""
        if 0 <= index < len(KeypointConverter.MHR70_NAMES):
            return KeypointConverter.MHR70_NAMES[index]
        raise IndexError(f"MHR70 index out of range: {index}")

    @staticmethod
    def get_mhr70_index(name: str) -> int:
        """Get MHR70 keypoint index by name."""
        try:
            return KeypointConverter.MHR70_NAMES.index(name)
        except ValueError:
            raise ValueError(f"Unknown MHR70 keypoint name: {name}")
