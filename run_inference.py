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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_stage import run_inference_stage
from utils.io_utils import get_output_dir
from utils.pipeline_options import add_inference_runtime_args, load_cli_defaults_from_argv


def parse_args(argv=None):
    defaults = load_cli_defaults_from_argv(argv, include_config=True)
    parser = argparse.ArgumentParser(description="SAM3D Body Inference")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output directory (default: auto)")
    add_inference_runtime_args(
        parser,
        defaults,
        include_device=True,
        include_config=True,
    )

    return parser.parse_args(argv)


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
    single_person: bool = True,
    support_surface_mode: str = None,
):
    """Run SAM3D Body inference and save to JSON."""
    stage = run_inference_stage(
        input_path=input_path,
        output_dir=output_dir,
        fps=fps,
        device=device,
        config_path=config_path,
        detector=detector,
        segmentor=segmentor,
        fov=fov,
        use_mask=use_mask,
        single_person=single_person,
        support_surface_mode=support_surface_mode,
        save_artifacts=True,
        save_raw_keypoints=False,
        show_header=True,
        header_title="SAM3D Body Inference",
        step_offset=0,
        step_total=2,
    )

    elapsed = stage["inference_meta"].get("inference_time", 0.0)
    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    throughput = len(stage["frame_paths"]) / elapsed if elapsed > 0 else 0.0
    print(f"Time: {elapsed:.1f}s ({throughput:.1f} FPS)")
    print(f"Frames: {len(stage['frame_paths'])}")
    print(f"Output: {stage['saved_paths']['video_outputs']}")
    print(f"{'='*60}\n")

    return str(stage["saved_paths"]["video_outputs"])


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
        single_person=args.single_person,
        support_surface_mode=args.support_surface_mode,
    )


if __name__ == "__main__":
    main()
