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

from sam3d_opensim.batch import collect_video_inputs, is_batch_input, resolve_video_output_dir
from src.subprocess_pipeline_runner import run_subprocess_pipeline
from utils.pipeline_options import (
    add_boolean_arg,
    add_inference_runtime_args,
    add_processing_args,
    add_subject_args,
    load_cli_defaults_from_argv,
)

PROJECT_ROOT = Path(__file__).parent


def parse_args(argv=None):
    defaults = load_cli_defaults_from_argv(argv, include_config=True)
    parser = argparse.ArgumentParser(description="Full SAM3D Body to OpenSim Pipeline")
    parser.add_argument(
        "--input",
        "-i",
        default=defaults["input_video_path"],
        help="Input video file or folder (defaults to input.video_path in config)",
    )
    add_subject_args(parser, defaults)
    parser.add_argument(
        "--output",
        "-o",
        default=defaults["output_dir"],
        help="Output directory (defaults to output.directory in config)",
    )
    add_boolean_arg(
        parser,
        "--skip-inference",
        default=defaults["skip_inference"],
        help_text="Skip SAM3D inference (use existing)",
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
    add_boolean_arg(
        parser,
        "--global-translation",
        default=defaults["global_translation"],
        help_text="Track global movement from cam_t",
    )
    add_inference_runtime_args(
        parser,
        defaults,
        include_device=False,
        include_config=True,
    )
    add_processing_args(parser, defaults)

    return parser.parse_args(argv)

def main():
    args = parse_args()

    if not args.input and not args.skip_inference:
        print("Pipeline ERROR: Input not provided. Set --input or input.video_path in Config.toml.")
        sys.exit(1)

    try:
        if args.skip_inference:
            run_subprocess_pipeline(
                project_root=PROJECT_ROOT,
                current_python=sys.executable,
                input_path=args.input,
                config_path=args.config,
                subject_height=args.height,
                subject_mass=args.mass,
                fps=args.fps,
                detector=args.detector,
                segmentor=args.segmentor,
                fov=args.fov,
                use_mask=args.use_mask,
                single_person=args.single_person,
                support_surface_mode=args.support_surface_mode,
                save_mesh_video=args.save_mesh_video,
                save_mesh_sequence=args.save_mesh_sequence,
                mesh_sequence_format=args.mesh_sequence_format,
                smooth=args.smooth,
                ground_alignment_mode=args.ground_alignment_mode,
                vertical_translation_mode=args.vertical_translation_mode,
                post_ik_foot_snap_mode=args.post_ik_foot_snap,
                ik_backend=args.ik_backend,
                save_graph=args.save_graph,
                global_translation=args.global_translation,
                skip_inference=args.skip_inference,
                skip_ik=args.skip_ik,
                skip_fbx=args.skip_fbx,
                output_dir=args.output,
            )
            return

        input_paths = collect_video_inputs(args.input)
        batch_mode = is_batch_input(args.input)
        for input_path in input_paths:
            output_dir = resolve_video_output_dir(
                video_path=input_path,
                configured_output_dir=args.output,
                batch_mode=batch_mode,
            )
            if batch_mode:
                print(f"\n[BATCH] Processing {input_path.name}")

            run_subprocess_pipeline(
                project_root=PROJECT_ROOT,
                current_python=sys.executable,
                input_path=str(input_path),
                config_path=args.config,
                subject_height=args.height,
                subject_mass=args.mass,
                fps=args.fps,
                detector=args.detector,
                segmentor=args.segmentor,
                fov=args.fov,
                use_mask=args.use_mask,
                single_person=args.single_person,
                support_surface_mode=args.support_surface_mode,
                save_mesh_video=args.save_mesh_video,
                save_mesh_sequence=args.save_mesh_sequence,
                mesh_sequence_format=args.mesh_sequence_format,
                smooth=args.smooth,
                ground_alignment_mode=args.ground_alignment_mode,
                vertical_translation_mode=args.vertical_translation_mode,
                post_ik_foot_snap_mode=args.post_ik_foot_snap,
                ik_backend=args.ik_backend,
                save_graph=args.save_graph,
                global_translation=args.global_translation,
                skip_inference=args.skip_inference,
                skip_ik=args.skip_ik,
                skip_fbx=args.skip_fbx,
                output_dir=str(output_dir),
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Pipeline ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Pipeline ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
