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

from sam3d_opensim.batch import collect_video_inputs, is_batch_input, resolve_video_output_dir
from src.inference_stage import run_inference_stage
from utils.pipeline_options import add_inference_runtime_args, load_cli_defaults_from_argv


def parse_args(argv=None):
    defaults = load_cli_defaults_from_argv(argv, include_config=True)
    parser = argparse.ArgumentParser(description="SAM3D Body Inference")
    parser.add_argument(
        "--input",
        "-i",
        default=defaults["input_video_path"],
        help="Input video file or folder (defaults to input.video_path in config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=defaults["output_dir"],
        help="Output directory (default: auto or output.directory in config)",
    )
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
    save_mesh_video: bool = False,
    save_mesh_sequence: bool = False,
    mesh_sequence_format: str = "ply",
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
        save_mesh_video=save_mesh_video,
        save_mesh_sequence=save_mesh_sequence,
        mesh_sequence_format=mesh_sequence_format,
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
    if stage["saved_paths"].get("mesh_video"):
        print(f"Mesh video: {stage['saved_paths']['mesh_video']}")
    if stage["saved_paths"].get("mesh_sequence_dir"):
        print(
            "Mesh sequence: "
            f"{stage['saved_paths']['mesh_sequence_dir']} "
            f"({stage['saved_paths'].get('mesh_sequence_count', 0)} files, "
            f"format={stage['saved_paths'].get('mesh_sequence_format')})"
        )
    print(f"{'='*60}\n")

    return str(stage["saved_paths"]["video_outputs"])


def main():
    args = parse_args()

    if not args.input:
        print("Error: Input not provided. Set --input or input.video_path in Config.toml.")
        sys.exit(1)

    try:
        input_paths = collect_video_inputs(args.input)
        batch_mode = is_batch_input(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    for input_path in input_paths:
        output_dir = resolve_video_output_dir(
            video_path=input_path,
            configured_output_dir=args.output,
            batch_mode=batch_mode,
        )
        if batch_mode:
            print(f"\n[BATCH] Processing {input_path.name}")

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
            save_mesh_video=args.save_mesh_video,
            save_mesh_sequence=args.save_mesh_sequence,
            mesh_sequence_format=args.mesh_sequence_format,
        )


if __name__ == "__main__":
    main()
