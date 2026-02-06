"""
SAM3D Body inference wrapper for video processing.

This module provides a high-level interface to run SAM3D Body inference
on video frames and extract MHR70 keypoints.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm


class SAM3DInference:
    """
    Wrapper for SAM3D Body model inference.

    Handles model loading, frame processing, and result extraction.
    """

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
        """
        self.sam3d_root = Path(sam3d_root)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bbox_threshold = bbox_threshold
        self.nms_threshold = nms_threshold
        self.use_mask = use_mask
        self.inference_type = inference_type

        # Add SAM3D Body to path
        if str(self.sam3d_root) not in sys.path:
            sys.path.insert(0, str(self.sam3d_root))

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

    def process_frame(
        self,
        image: np.ndarray,
        frame_idx: int = 0,
        bboxes: Optional[np.ndarray] = None,
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
        outputs = self.estimator.process_one_image(
            image,
            bboxes=bboxes,
            bbox_thr=self.bbox_threshold,
            nms_thr=self.nms_threshold,
            use_mask=self.use_mask,
            inference_type=self.inference_type,
        )

        # Cache focal length from first frame
        if frame_idx == 0 and outputs:
            if "focal_length" in outputs[0]:
                self._cached_focal_length = outputs[0]["focal_length"]
            # Disable FOV estimator for subsequent frames (expensive)
            if (
                hasattr(self.estimator, "fov_estimator")
                and self.estimator.fov_estimator is not None
            ):
                self.estimator.fov_estimator = None
        elif self._cached_focal_length is not None:
            # Apply cached focal length to subsequent frames
            for out in outputs:
                out["focal_length"] = self._cached_focal_length

        return outputs

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
                - fps: Frame rate (if available)
                - num_frames: Number of frames processed
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
            outputs = self.process_frame(image, frame_idx=idx)

            # Take first person only (single-person pipeline)
            if outputs:
                all_outputs.append(
                    {
                        "frame_idx": idx,
                        "frame_path": frame_path,
                        "output": outputs[0],  # First person
                    }
                )
            else:
                # No detection - store None
                all_outputs.append(
                    {
                        "frame_idx": idx,
                        "frame_path": frame_path,
                        "output": None,
                    }
                )

        return {
            "frames": all_outputs,
            "num_frames": len(all_outputs),
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
