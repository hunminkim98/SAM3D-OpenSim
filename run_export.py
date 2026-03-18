#!/usr/bin/env python3
"""
Export SAM3D Body Results to OpenSim
====================================

Reads video_outputs.json and exports to TRC/MOT/FBX.
This is fast - iterate on settings without re-running inference.

Usage:
    # From our inference output
    python run_export.py --input output_dir/video_outputs.json --height 1.69

    # From original SAM3D Body output
    python run_export.py --input C:/Sam3dBody/sam-3d-body/outputs/my_video/video_outputs.json --height 1.75

    # With global translation (for walking videos)
    python run_export.py --input video_outputs.json --height 1.69 --global-translation

    # Skip IK, only generate TRC
    python run_export.py --input video_outputs.json --height 1.69 --skip-ik
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.export_stage import run_export_stage
from utils.pipeline_options import (
    add_boolean_arg,
    add_processing_args,
    add_subject_args,
    load_cli_defaults_from_argv,
)


def parse_args(argv=None):
    defaults = load_cli_defaults_from_argv(argv, include_config=True)
    parser = argparse.ArgumentParser(description="Export SAM3D outputs to OpenSim")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument(
        "--input",
        "-i",
        default=defaults["input_video_outputs_path"],
        help="video_outputs.json file (defaults to input.video_outputs_path in config)",
    )
    add_subject_args(parser, defaults)
    parser.add_argument(
        "--output",
        "-o",
        default=defaults["output_dir"],
        help="Output directory (default: same as input or output.directory in config)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS override (default: from metadata)",
    )
    add_boolean_arg(
        parser,
        "--global-translation",
        default=defaults["global_translation"],
        help_text="Track global movement",
    )
    add_boolean_arg(
        parser,
        "--skip-ik",
        default=defaults["skip_ik"],
        help_text="Skip OpenSim IK",
    )
    add_boolean_arg(
        parser,
        "--skip-fbx",
        default=defaults["skip_fbx"],
        help_text="Skip FBX export",
    )
    parser.add_argument(
        "--person",
        type=int,
        default=defaults["person_idx"],
        help="Person index if multiple detected",
    )
    add_processing_args(parser, defaults)
    return parser.parse_args(argv)

def run_export(
    json_path: str,
    output_dir: str,
    subject_height: float,
    subject_mass: float,
    fps: float,
    global_translation: bool,
    skip_ik: bool,
    skip_fbx: bool,
    person_idx: int,
    smooth_cutoff: float = 6.0,
    ground_alignment_mode: str = "auto",
    vertical_translation_mode: str = "auto",
    post_ik_foot_snap_mode: str = "off",
    ik_backend: str = "direct_opensim",
    save_graph: bool = False,
):
    """Export SAM3D outputs to TRC/MOT/FBX."""
    start_time = time.time()
    results = run_export_stage(
        json_path=json_path,
        output_dir=output_dir,
        subject_height=subject_height,
        subject_mass=subject_mass,
        fps=fps,
        global_translation=global_translation,
        skip_ik=skip_ik,
        skip_fbx=skip_fbx,
        person_idx=person_idx,
        smooth_cutoff=smooth_cutoff,
        ground_alignment_mode=ground_alignment_mode,
        vertical_translation_mode=vertical_translation_mode,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        ik_backend=ik_backend,
        save_graph=save_graph,
        show_header=True,
        header_title="SAM3D Body Export to OpenSim",
        step_offset=0,
        step_total=5,
        project_root=PROJECT_ROOT,
    )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nOutput files:")
    for name, path in results.items():
        if name in {"ground_alignment", "post_ik_contact_meta", "elapsed", "ik_backend"}:
            continue
        status = "OK" if path and Path(path).exists() else "SKIPPED"
        print(f"  [{status}] {name.upper()}: {path}")
    print(f"{'='*60}\n")

    return results

def main():
    args = parse_args()

    if not args.input:
        print(
            "Error: Input not provided. Set --input or input.video_outputs_path in Config.toml."
        )
        sys.exit(1)

    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: Input not found: {json_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = json_path.parent

    run_export(
        json_path=str(json_path),
        output_dir=str(output_dir),
        subject_height=args.height,
        subject_mass=args.mass,
        fps=args.fps,
        global_translation=args.global_translation,
        skip_ik=args.skip_ik,
        skip_fbx=args.skip_fbx,
        person_idx=args.person,
        smooth_cutoff=args.smooth,
        ground_alignment_mode=args.ground_alignment_mode,
        vertical_translation_mode=args.vertical_translation_mode,
        post_ik_foot_snap_mode=args.post_ik_foot_snap,
        ik_backend=args.ik_backend,
        save_graph=args.save_graph,
    )


if __name__ == "__main__":
    main()
