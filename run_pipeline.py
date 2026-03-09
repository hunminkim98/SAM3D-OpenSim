#!/usr/bin/env python3
"""
SAM3D Body to OpenSim Pipeline
==============================

Main entry point for converting video to OpenSim motion data.

Usage:
    python run_pipeline.py --input video.mp4 --height 1.75

The pipeline:
1. Extracts frames from video
2. Runs SAM3D Body inference to get 3D pose
3. Converts MHR70 keypoints to OpenSim markers
4. Transforms coordinates from camera to OpenSim frame
5. Exports TRC marker file
6. Runs OpenSim inverse kinematics to get joint angles
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.video_utils import extract_frames, get_video_info
from utils.io_utils import load_config, load_marker_mapping, save_json, get_output_dir
from utils.cli_utils import str_to_bool


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SAM3D Body to OpenSim Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input video file path",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.75,
        help="Subject height in meters (default: 1.75)",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=70.0,
        help="Subject mass in kg (default: 70.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target FPS for frame extraction (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip inverse kinematics (only generate TRC)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization outputs",
    )
    parser.add_argument(
        "--save-fbx",
        action="store_true",
        help="Export FBX skeleton animation (requires Blender)",
    )
    parser.add_argument(
        "--global-translation",
        action="store_true",
        help="Apply global translation from cam_t (track walking movement)",
    )

    # SAM3D Body component options
    parser.add_argument("--detector", default="vitdet", choices=["vitdet", "yolo11", "sam3", "none"],
                        help="Human detector: vitdet, yolo11, sam3, or none (default: vitdet)")
    parser.add_argument("--segmentor", default=None, choices=["sam2", "none"],
                        help="Segmentor: sam2 or none (default: none)")
    parser.add_argument("--fov", default="moge2", choices=["moge2", "none"],
                        help="FOV estimator: moge2 or none (default: moge2)")
    parser.add_argument("--use-mask", action="store_true",
                        help="Use segmentation mask (requires segmentor)")
    parser.add_argument("--smooth", type=float, default=6.0,
                        help="Smoothing cutoff frequency in Hz (0 to disable, default: 6.0)")
    parser.add_argument(
        "--ground-alignment-mode",
        type=str,
        choices=["auto", "contact_aware", "per_frame_snap"],
        default=None,
        help="Ground alignment strategy (default: from config, usually auto)",
    )
    parser.add_argument(
        "--single_person",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Prompt once to choose a single tracked person (default: true)",
    )

    return parser.parse_args()


def run_pipeline(
    input_path: str,
    subject_height: float = 1.75,
    subject_mass: float = 70.0,
    output_dir: Optional[str] = None,
    target_fps: float = 30.0,
    device: str = "cuda",
    config_path: Optional[str] = None,
    skip_ik: bool = False,
    visualize: bool = False,
    save_fbx: bool = False,
    global_translation: bool = False,
    detector: str = "vitdet",
    segmentor: Optional[str] = None,
    fov: str = "moge2",
    use_mask: bool = False,
    smooth_cutoff: float = 6.0,
    ground_alignment_mode: Optional[str] = None,
    single_person: bool = True,
) -> dict:
    """
    Run the full SAM3D Body to OpenSim pipeline.

    Args:
        input_path: Path to input video
        subject_height: Subject height in meters
        subject_mass: Subject mass in kg
        output_dir: Output directory
        target_fps: Target FPS for processing
        device: Inference device ('cuda' or 'cpu')
        config_path: Path to config file
        skip_ik: Whether to skip inverse kinematics
        visualize: Whether to generate visualizations
        save_fbx: Whether to export FBX animation
        global_translation: Whether to apply global translation from cam_t
        detector: Human detector ('vitdet', 'sam3', or 'none')
        segmentor: Segmentor ('sam2' or None)
        fov: FOV estimator ('moge2' or 'none')
        use_mask: Whether to use segmentation masks
        ground_alignment_mode: Ground alignment mode override
        single_person: Whether to prompt once and track a single chosen person

    Returns:
        Dictionary with pipeline results and output paths
    """
    start_time = time.time()
    results = {"success": False, "outputs": {}, "timings": {}}

    # Load configuration
    config = load_config(config_path)
    marker_mapping = load_marker_mapping()
    resolved_ground_alignment_mode = (
        ground_alignment_mode
        or config.get("processing", {}).get("ground_alignment_mode", "auto")
    )

    # Setup output directory
    if output_dir is None:
        output_dir = get_output_dir(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results["outputs"]["directory"] = str(output_dir)

    # Parse SAM3D component options
    detector_name = detector if detector != "none" else None
    segmentor_name = segmentor if segmentor and segmentor != "none" else None
    fov_name = fov if fov != "none" else None

    print(f"\n{'='*60}")
    print("SAM3D Body to OpenSim Pipeline")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
    print(f"Device: {device}")
    print(f"Detector: {detector_name or 'none'}, FOV: {fov_name or 'none'}, Segmentor: {segmentor_name or 'none'}")
    print(f"Ground alignment: {resolved_ground_alignment_mode}")
    print(f"Single-person selection: {'ENABLED' if single_person else 'disabled'}")
    print(f"{'='*60}\n")

    # Step 1: Get video info and extract frames
    print("[1/6] Extracting frames from video...")
    t0 = time.time()

    video_info = get_video_info(input_path)
    print(f"  Video: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s")

    frames_dir = output_dir / "frames"
    frame_paths, actual_fps = extract_frames(
        input_path, str(frames_dir), target_fps=target_fps
    )
    results["timings"]["frame_extraction"] = time.time() - t0
    print(f"  Extracted {len(frame_paths)} frames at {actual_fps:.2f} FPS")

    # Step 2: Run SAM3D Body inference
    print("\n[2/6] Running SAM3D Body inference...")
    t0 = time.time()

    from src.sam3d_inference import SAM3DInference

    sam3d = SAM3DInference(
        sam3d_root=config["sam3d"]["sam3d_root"],
        checkpoint_path=config["sam3d"]["checkpoint"],
        mhr_path=config["sam3d"]["mhr_path"],
        device=device,
        detector_name=detector_name,
        detector_path=config["sam3d"].get("detector_path"),
        segmentor_name=segmentor_name,
        segmentor_path=config["sam3d"].get("segmentor_path"),
        fov_name=fov_name,
        fov_path=config["sam3d"].get("fov_path"),
        bbox_threshold=config["sam3d"]["bbox_threshold"],
        use_mask=use_mask,
        inference_type=config["sam3d"]["inference_type"],
        single_person=single_person,
    )

    inference_results = sam3d.process_video(frame_paths, progress=True)
    keypoints_3d, valid_frames = sam3d.extract_keypoints_3d(inference_results["frames"])
    camera_params = sam3d.extract_camera_params(inference_results["frames"])

    results["timings"]["sam3d_inference"] = time.time() - t0
    print(f"  Processed {inference_results['num_frames']} frames, "
          f"{np.sum(valid_frames)} valid detections")

    # Save raw keypoints (legacy format)
    raw_json_path = output_dir / "keypoints_raw.json"
    save_json(
        {
            "keypoints_3d": keypoints_3d.tolist(),
            "valid_frames": valid_frames.tolist(),
            "camera_params": {k: v.tolist() for k, v in camera_params.items()},
            "fps": actual_fps,
        },
        raw_json_path,
    )
    results["outputs"]["raw_keypoints"] = str(raw_json_path)

    # Save in SAM3D format (video_outputs.json) for use with run_export.py
    sam3d_outputs = []
    for i, frame_data in enumerate(inference_results["frames"]):
        frame_name = Path(frame_data["frame_path"]).name
        frame_entry = {"frame": frame_name, "outputs": []}

        outputs_to_save = []
        if single_person:
            if frame_data["output"] is not None:
                outputs_to_save = [frame_data["output"]]
        else:
            outputs_to_save = frame_data.get("outputs", [])

        for output_idx, out in enumerate(outputs_to_save):
            if single_person and output_idx == 0:
                pred_keypoints_3d = keypoints_3d[i].tolist()
                pred_cam_t = camera_params["cam_translations"][i].tolist()
            else:
                pred_keypoints_3d = np.array(out.get("pred_keypoints_3d", [])).tolist()
                pred_cam_t = np.array(out.get("pred_cam_t", [0, 0, 5])).tolist()

            person_data = {
                "bbox": out.get("bbox", [0, 0, video_info['width'], video_info['height']]),
                "focal_length": float(out.get("focal_length", 1000.0)),
                "pred_keypoints_3d": pred_keypoints_3d,
                "pred_cam_t": pred_cam_t,
            }
            if "pred_keypoints_2d" in out:
                person_data["pred_keypoints_2d"] = np.array(out["pred_keypoints_2d"]).tolist()
            if "shape_params" in out:
                person_data["shape_params"] = np.array(out["shape_params"]).tolist()

            frame_entry["outputs"].append(person_data)

        sam3d_outputs.append(frame_entry)

    sam3d_json_path = output_dir / "video_outputs.json"
    save_json(sam3d_outputs, sam3d_json_path)
    results["outputs"]["video_outputs"] = str(sam3d_json_path)

    # Save metadata for run_export.py
    meta_path = output_dir / "inference_meta.json"
    meta_payload = {
        "input_video": str(input_path),
        "fps": actual_fps,
        "num_frames": len(frame_paths),
        "video_info": video_info,
        "single_person": single_person,
        "selection": inference_results.get("selection", {}),
    }

    print(f"  Saved SAM3D format: {sam3d_json_path}")

    # Step 3: Post-process keypoints
    print("\n[3/6] Post-processing keypoints...")
    t0 = time.time()

    from src.post_processing import PostProcessor
    from src.coordinate_transform import CoordinateTransformer

    # Apply post-processing
    use_smoothing = smooth_cutoff > 0
    if use_smoothing:
        print(f"  Smoothing enabled: {smooth_cutoff} Hz cutoff")
    post_processor = PostProcessor(
        smooth_filter=use_smoothing,
        filter_cutoff=smooth_cutoff,
        normalize_bones=config["processing"]["normalize_bones"],
    )
    keypoints_processed = post_processor.process(
        keypoints_3d, fps=actual_fps, subject_height=subject_height
    )

    # Transform coordinates
    transformer = CoordinateTransformer(
        subject_height=subject_height,
        units="mm",  # TRC uses mm
    )

    # Get camera translations if using global translation
    cam_translations = None
    if global_translation:
        cam_translations = camera_params.get("cam_translations")
        if cam_translations is not None:
            print(f"  Using global translation from cam_t ({cam_translations.shape[0]} frames)")

    keypoints_opensim = transformer.transform(
        keypoints_processed,
        camera_translation=cam_translations,
        center_pelvis=not global_translation,  # Don't center if using global translation
        align_to_ground=True,
        apply_global_translation=global_translation,
        ground_alignment_mode=resolved_ground_alignment_mode,
    )
    ground_alignment_info = transformer.get_last_ground_alignment_info()
    print(
        "  Ground alignment applied: "
        f"{ground_alignment_info.get('applied_mode')} "
        f"(contact_frames={ground_alignment_info.get('contact_frames')}, "
        f"flight_frames={ground_alignment_info.get('flight_frames')})"
    )
    meta_payload["ground_alignment"] = ground_alignment_info
    save_json(meta_payload, meta_path)

    # Correct forward lean if enabled
    if config["processing"]["correct_forward_lean"]:
        keypoints_opensim = transformer.correct_forward_lean(keypoints_opensim)

    results["timings"]["post_processing"] = time.time() - t0
    print("  Applied coordinate transformation and normalization")

    # Step 4: Convert to OpenSim markers
    print("\n[4/6] Converting to OpenSim markers...")
    t0 = time.time()

    from src.keypoint_converter import KeypointConverter

    converter = KeypointConverter(
        mapping_path=str(PROJECT_ROOT / "config" / "marker_mapping.yaml")
    )
    markers, marker_names = converter.convert(keypoints_opensim, include_derived=True)

    results["timings"]["marker_conversion"] = time.time() - t0
    print(f"  Generated {len(marker_names)} markers")

    # Step 5: Export TRC file
    print("\n[5/6] Exporting TRC file...")
    t0 = time.time()

    from src.trc_exporter import TRCExporter

    trc_exporter = TRCExporter(fps=actual_fps, units="mm")
    video_name = Path(input_path).stem
    trc_path = output_dir / f"markers_{video_name}.trc"
    trc_exporter.export(markers, marker_names, str(trc_path))

    results["outputs"]["trc"] = str(trc_path)
    results["timings"]["trc_export"] = time.time() - t0
    print(f"  Saved: {trc_path}")

    # Step 6: Run inverse kinematics (optional)
    if not skip_ik:
        print("\n[6/6] Running inverse kinematics...")
        t0 = time.time()

        from src.opensim_ik import OpenSimIK

        model_path = PROJECT_ROOT / config["opensim"]["model"]
        markers_xml_path = PROJECT_ROOT / config["opensim"]["markers_xml"]

        try:
            ik_solver = OpenSimIK(
                model_path=str(model_path),
                markers_xml_path=str(markers_xml_path),
                accuracy=config["opensim"]["ik_accuracy"],
            )

            ik_results = ik_solver.run_ik(
                trc_path=str(trc_path),
                output_dir=str(output_dir),
                subject_height=subject_height,
                subject_mass=subject_mass,
            )

            results["outputs"]["mot"] = ik_results.get("mot")
            results["outputs"]["scaled_model"] = ik_results.get("scaled_model")
            results["timings"]["inverse_kinematics"] = time.time() - t0
            print(f"  Saved: {ik_results.get('mot')}")

        except Exception as e:
            print(f"  Warning: IK failed: {e}")
            results["outputs"]["mot"] = None
            results["timings"]["inverse_kinematics"] = time.time() - t0
    else:
        print("\n[6/6] Skipping inverse kinematics (--skip-ik)")

    # Calculate total time
    total_time = time.time() - start_time
    results["timings"]["total"] = total_time
    results["success"] = True

    # Generate report
    report_path = output_dir / "processing_report.json"
    save_json(
        {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "subject": {
                "height": subject_height,
                "mass": subject_mass,
            },
            "video_info": video_info,
            "processing": {
                "fps": actual_fps,
                "num_frames": len(frame_paths),
                "valid_frames": int(np.sum(valid_frames)),
                "num_markers": len(marker_names),
                "single_person": single_person,
                "ground_alignment": ground_alignment_info,
            },
            "timings": results["timings"],
            "outputs": results["outputs"],
            "selection": inference_results.get("selection", {}),
        },
        report_path,
    )
    results["outputs"]["report"] = str(report_path)

    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Frames processed: {len(frame_paths)}")
    print(f"Valid detections: {np.sum(valid_frames)}")
    print(f"\nOutputs:")
    for key, path in results["outputs"].items():
        if path:
            print(f"  {key}: {path}")
    print(f"{'='*60}\n")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run pipeline
    try:
        results = run_pipeline(
            input_path=args.input,
            subject_height=args.height,
            subject_mass=args.mass,
            output_dir=args.output,
            target_fps=args.fps,
            device=args.device,
            config_path=args.config,
            skip_ik=args.skip_ik,
            visualize=args.visualize,
            save_fbx=args.save_fbx,
            global_translation=args.global_translation,
            detector=args.detector,
            segmentor=args.segmentor,
            fov=args.fov,
            use_mask=args.use_mask,
            smooth_cutoff=args.smooth,
            ground_alignment_mode=args.ground_alignment_mode,
            single_person=args.single_person,
        )

        if not results["success"]:
            print("Pipeline failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
