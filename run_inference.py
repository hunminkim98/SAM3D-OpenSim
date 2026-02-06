#!/usr/bin/env python3
"""
SAM3D Body Inference Only
=========================

Runs SAM3D Body inference and saves results to video_outputs.json.
This is the slow step - run once, then iterate on export settings.

Usage:
    python run_inference.py --input video.mp4
    python run_inference.py --input video.mp4 --output my_output_dir

Output:
    output_dir/
    ├── frames/               # Extracted video frames
    └── video_outputs.json    # SAM3D Body outputs (keypoints, cam_t, etc.)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import extract_frames, get_video_info
from utils.io_utils import load_config, get_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3D Body Inference")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output directory (default: auto)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--config", help="Config file path")

    # SAM3D Body component options
    parser.add_argument("--detector", default="vitdet", choices=["vitdet", "yolo11", "sam3", "none"],
                        help="Human detector: vitdet, yolo11, sam3, or none (default: vitdet)")
    parser.add_argument("--segmentor", default=None, choices=["sam2", "none"],
                        help="Segmentor: sam2 or none (default: none)")
    parser.add_argument("--fov", default="moge2", choices=["moge2", "none"],
                        help="FOV estimator: moge2 or none (default: moge2)")
    parser.add_argument("--use-mask", action="store_true",
                        help="Use segmentation mask (requires segmentor)")

    return parser.parse_args()


def run_inference(
    input_path: str,
    output_dir: str,
    fps: float,
    device: str,
    config_path: str = None,
    detector: str = "vitdet",
    segmentor: str = None,
    fov: str = "moge2",
    use_mask: bool = False,
):
    """Run SAM3D Body inference and save to JSON."""
    start_time = time.time()

    # Load config
    config = load_config(config_path)

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse component options
    detector_name = detector if detector != "none" else None
    segmentor_name = segmentor if segmentor and segmentor != "none" else None
    fov_name = fov if fov != "none" else None

    print(f"\n{'='*60}")
    print("SAM3D Body Inference")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Detector: {detector_name or 'none'}")
    print(f"Segmentor: {segmentor_name or 'none'}")
    print(f"FOV estimator: {fov_name or 'none'}")
    print(f"Use mask: {use_mask}")
    print(f"{'='*60}\n")

    # Step 1: Extract frames
    print("[1/2] Extracting frames...")
    video_info = get_video_info(input_path)
    print(f"  Video: {video_info['width']}x{video_info['height']}, {video_info['fps']:.2f} FPS")

    frames_dir = output_dir / "frames"
    frame_paths, actual_fps = extract_frames(input_path, str(frames_dir), target_fps=fps)
    print(f"  Extracted {len(frame_paths)} frames at {actual_fps:.2f} FPS")

    # Step 2: Run SAM3D Body
    print("\n[2/2] Running SAM3D Body inference...")

    from src.sam3d_inference import SAM3DInference

    sam3d = SAM3DInference(
        sam3d_root=config["sam3d"]["sam3d_root"],
        checkpoint_path=config["sam3d"]["checkpoint"],
        mhr_path=config["sam3d"]["mhr_path"],
        device=device,
        detector_name=detector_name,
        segmentor_name=segmentor_name,
        fov_name=fov_name,
        bbox_threshold=config["sam3d"]["bbox_threshold"],
        use_mask=use_mask,
        inference_type=config["sam3d"]["inference_type"],
    )

    # Process frames and collect outputs in SAM3D format
    all_outputs = []

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing")):
        image = cv2.imread(frame_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputs = sam3d.process_frame(image, frame_idx=idx)

        frame_name = Path(frame_path).name
        frame_data = {
            "frame": frame_name,
            "outputs": []
        }

        for out in outputs:
            person_data = {
                "bbox": out.get("bbox", [0, 0, video_info['width'], video_info['height']]),
                "focal_length": float(out.get("focal_length", 1000.0)),
                "pred_keypoints_3d": np.array(out.get("pred_keypoints_3d", [])).tolist(),
                "pred_cam_t": np.array(out.get("pred_cam_t", [0, 0, 5])).tolist(),
            }
            # Include optional fields if present
            if "pred_keypoints_2d" in out:
                person_data["pred_keypoints_2d"] = np.array(out["pred_keypoints_2d"]).tolist()
            if "shape_params" in out:
                person_data["shape_params"] = np.array(out["shape_params"]).tolist()

            frame_data["outputs"].append(person_data)

        all_outputs.append(frame_data)

    # Save to JSON (SAM3D format)
    json_path = output_dir / "video_outputs.json"
    with open(json_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)

    # Also save metadata
    meta_path = output_dir / "inference_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "input_video": str(input_path),
            "fps": actual_fps,
            "num_frames": len(frame_paths),
            "video_info": video_info,
            "inference_time": time.time() - start_time,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s ({len(frame_paths)/elapsed:.1f} FPS)")
    print(f"Frames: {len(frame_paths)}")
    print(f"Output: {json_path}")
    print(f"{'='*60}\n")

    return str(json_path)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = get_output_dir(str(input_path))

    run_inference(
        input_path=str(input_path),
        output_dir=str(output_dir),
        fps=args.fps,
        device=args.device,
        config_path=args.config,
        detector=args.detector,
        segmentor=args.segmentor,
        fov=args.fov,
        use_mask=args.use_mask,
    )


if __name__ == "__main__":
    main()
