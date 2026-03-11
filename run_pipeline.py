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
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_runner import run_combined_pipeline
from utils.pipeline_options import (
    add_inference_runtime_args,
    add_processing_args,
    add_subject_args,
    load_cli_defaults_from_argv,
)


def parse_args(argv=None):
    """Parse command line arguments."""
    defaults = load_cli_defaults_from_argv(argv, include_config=True)
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
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    add_subject_args(parser, defaults)
    add_inference_runtime_args(
        parser,
        defaults,
        include_device=True,
        include_config=True,
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip inverse kinematics (only generate TRC)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Deprecated no-op flag kept for compatibility",
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
    add_processing_args(parser, defaults)

    return parser.parse_args(argv)


def run_pipeline(
    input_path: str,
    subject_height: Optional[float] = None,
    subject_mass: Optional[float] = None,
    output_dir: Optional[str] = None,
    target_fps: Optional[float] = None,
    device: Optional[str] = None,
    config_path: Optional[str] = None,
    skip_ik: bool = False,
    visualize: bool = False,
    save_fbx: bool = False,
    global_translation: bool = False,
    detector: Optional[str] = None,
    segmentor: Optional[str] = None,
    fov: Optional[str] = None,
    use_mask: bool = False,
    smooth_cutoff: Optional[float] = None,
    ground_alignment_mode: Optional[str] = None,
    vertical_translation_mode: Optional[str] = None,
    single_person: Optional[bool] = None,
    support_surface_mode: Optional[str] = None,
    post_ik_foot_snap_mode: Optional[str] = None,
    save_graph: bool = False,
) -> dict:
    """Backward-compatible wrapper over the shared in-process pipeline runner."""
    return run_combined_pipeline(
        input_path=input_path,
        project_root=PROJECT_ROOT,
        subject_height=subject_height,
        subject_mass=subject_mass,
        output_dir=output_dir,
        target_fps=target_fps,
        device=device,
        config_path=config_path,
        skip_ik=skip_ik,
        visualize=visualize,
        save_fbx=save_fbx,
        global_translation=global_translation,
        detector=detector,
        segmentor=segmentor,
        fov=fov,
        use_mask=use_mask,
        smooth_cutoff=smooth_cutoff,
        ground_alignment_mode=ground_alignment_mode,
        vertical_translation_mode=vertical_translation_mode,
        single_person=single_person,
        support_surface_mode=support_surface_mode,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        save_graph=save_graph,
    )


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
            vertical_translation_mode=args.vertical_translation_mode,
            single_person=args.single_person,
            support_surface_mode=args.support_surface_mode,
            post_ik_foot_snap_mode=args.post_ik_foot_snap,
            save_graph=args.save_graph,
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
