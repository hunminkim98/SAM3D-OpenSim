#!/usr/bin/env python3
"""
Full SAM3D Body to OpenSim Pipeline Runner
==========================================

This script orchestrates the canonical two-stage pipeline:
1. SAM3D Body inference via run_inference.py in the sam_3d_body environment
2. Export / OpenSim IK / FBX via run_export.py in the current environment
"""

import argparse
import sys
from pathlib import Path

from src.subprocess_pipeline_runner import run_subprocess_pipeline
from utils.pipeline_options import (
    add_inference_runtime_args,
    add_processing_args,
    add_subject_args,
    load_cli_defaults_from_argv,
)

PROJECT_ROOT = Path(__file__).parent


def parse_args(argv=None):
    defaults = load_cli_defaults_from_argv(argv)
    parser = argparse.ArgumentParser(description="Full SAM3D Body to OpenSim Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    add_subject_args(parser, defaults)
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--skip-inference", action="store_true", help="Skip SAM3D inference (use existing)")
    parser.add_argument("--skip-ik", action="store_true", help="Skip OpenSim IK")
    parser.add_argument("--skip-fbx", action="store_true", help="Skip FBX export")
    parser.add_argument("--global-translation", action="store_true", help="Track global movement from cam_t")
    add_inference_runtime_args(
        parser,
        defaults,
        include_device=False,
        include_config=False,
    )
    add_processing_args(parser, defaults)

    return parser.parse_args(argv)

def main():
    args = parse_args()

    try:
        run_subprocess_pipeline(
            project_root=PROJECT_ROOT,
            current_python=sys.executable,
            input_path=args.input,
            subject_height=args.height,
            subject_mass=args.mass,
            fps=args.fps,
            detector=args.detector,
            segmentor=args.segmentor,
            fov=args.fov,
            use_mask=args.use_mask,
            single_person=args.single_person,
            support_surface_mode=args.support_surface_mode,
            smooth=args.smooth,
            ground_alignment_mode=args.ground_alignment_mode,
            vertical_translation_mode=args.vertical_translation_mode,
            post_ik_foot_snap_mode=args.post_ik_foot_snap,
            save_graph=args.save_graph,
            global_translation=args.global_translation,
            skip_inference=args.skip_inference,
            skip_ik=args.skip_ik,
            skip_fbx=args.skip_fbx,
            output_dir=args.output,
        )
    except Exception as exc:
        print(f"Pipeline ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
