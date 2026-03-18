"""Console entry point for the Config.toml-driven pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sam3d_opensim.batch import collect_video_inputs, is_batch_input, resolve_video_output_dir
from sam3d_opensim.config import project_root, runtime_workspace
from src.subprocess_pipeline_runner import run_subprocess_pipeline
from utils.pipeline_options import load_cli_defaults


def _resolve_mode(config_path: str | None, cli_mode: str | None) -> str:
    if cli_mode:
        return cli_mode
    return load_cli_defaults(config_path)["mode"]


def _require_path(value: str | None, message: str) -> str:
    if value:
        return value
    raise ValueError(message)


def _resolve_cli_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SAM3D-OpenSim console entry point")
    parser.add_argument("--config", help="Config.toml path")
    parser.add_argument(
        "--mode",
        choices=["full", "inference", "export", "pipeline"],
        help="Override run.mode from config",
    )
    parser.add_argument("--input", help="Override input.video_path")
    parser.add_argument("--video-outputs", help="Override input.video_outputs_path")
    parser.add_argument("--output", help="Override output.directory")
    args = parser.parse_args(argv)

    config_path = _resolve_cli_path(args.config)
    input_override = _resolve_cli_path(args.input)
    video_outputs_override = _resolve_cli_path(args.video_outputs)
    output_override = _resolve_cli_path(args.output)

    defaults = load_cli_defaults(config_path)
    mode = _resolve_mode(config_path, args.mode)
    root = project_root()
    workspace_root = runtime_workspace()

    if mode == "full":
        input_path = input_override or defaults["input_video_path"]
        output_dir = output_override or defaults["output_dir"]
        existing_outputs = video_outputs_override or defaults["input_video_outputs_path"]
        if defaults["skip_inference"] and output_dir is None and existing_outputs:
            output_dir = str(Path(existing_outputs).expanduser().resolve().parent)
        if not defaults["skip_inference"]:
            input_path = _require_path(
                input_path,
                "Input video not provided. Set --input or input.video_path in Config.toml.",
            )
            video_inputs = collect_video_inputs(input_path)
            batch_mode = is_batch_input(input_path)
        elif input_path is None and output_dir is None:
            raise ValueError(
                "skip_inference=true requires output.directory or input.video_outputs_path."
            )
        else:
            if input_override is None:
                input_path = None
            video_inputs = [None]
            batch_mode = False

        for video_input in video_inputs:
            resolved_output_dir = output_dir
            if video_input is not None:
                resolved_output_dir = str(
                    resolve_video_output_dir(
                        video_path=video_input,
                        configured_output_dir=output_dir,
                        batch_mode=batch_mode,
                    )
                )
                if batch_mode:
                    print(f"\n[BATCH] Processing {video_input.name}")
            run_subprocess_pipeline(
                project_root=root,
                workspace_root=workspace_root,
                current_python=sys.executable,
                input_path=str(video_input) if video_input is not None else input_path,
                config_path=config_path,
                subject_height=defaults["height"],
                subject_mass=defaults["mass"],
                fps=defaults["fps"],
                detector=defaults["detector"],
                segmentor=defaults["segmentor"],
                fov=defaults["fov"],
                use_mask=defaults["use_mask"],
                single_person=defaults["single_person"],
                support_surface_mode=defaults["support_surface_mode"],
                save_mesh_video=defaults["save_mesh_video"],
                save_mesh_sequence=defaults["save_mesh_sequence"],
                mesh_sequence_format=defaults["mesh_sequence_format"],
                smooth=defaults["smooth"],
                ground_alignment_mode=defaults["ground_alignment_mode"],
                vertical_translation_mode=defaults["vertical_translation_mode"],
                post_ik_foot_snap_mode=defaults["post_ik_foot_snap_mode"],
                ik_backend=defaults["ik_backend"],
                save_graph=defaults["save_graph"],
                global_translation=defaults["global_translation"],
                skip_inference=defaults["skip_inference"],
                skip_ik=defaults["skip_ik"],
                skip_fbx=defaults["skip_fbx"],
                output_dir=resolved_output_dir,
            )
        return

    if mode == "inference":
        from run_inference import run_inference

        input_path = _require_path(
            input_override or defaults["input_video_path"],
            "Input video not provided. Set --input or input.video_path in Config.toml.",
        )
        video_inputs = collect_video_inputs(input_path)
        batch_mode = is_batch_input(input_path)
        output_dir = output_override or defaults["output_dir"]
        for video_input in video_inputs:
            resolved_output_dir = str(
                resolve_video_output_dir(
                    video_path=video_input,
                    configured_output_dir=output_dir,
                    batch_mode=batch_mode,
                )
            )
            if batch_mode:
                print(f"\n[BATCH] Processing {video_input.name}")
            run_inference(
                input_path=str(video_input),
                output_dir=resolved_output_dir,
                fps=defaults["fps"],
                device=defaults["device"],
                config_path=config_path,
                detector=defaults["detector"],
                segmentor=defaults["segmentor"],
                fov=defaults["fov"],
                use_mask=defaults["use_mask"],
                single_person=defaults["single_person"],
                support_surface_mode=defaults["support_surface_mode"],
                save_mesh_video=defaults["save_mesh_video"],
                save_mesh_sequence=defaults["save_mesh_sequence"],
                mesh_sequence_format=defaults["mesh_sequence_format"],
            )
        return

    if mode == "export":
        from run_export import run_export

        json_path = _require_path(
            video_outputs_override or defaults["input_video_outputs_path"],
            "Export input not provided. Set --video-outputs or input.video_outputs_path in Config.toml.",
        )
        output_dir = output_override or defaults["output_dir"]
        if output_dir is None:
            output_dir = str(Path(json_path).parent)
        run_export(
            json_path=json_path,
            output_dir=output_dir,
            subject_height=defaults["height"],
            subject_mass=defaults["mass"],
            fps=defaults["fps"],
            global_translation=defaults["global_translation"],
            skip_ik=defaults["skip_ik"],
            skip_fbx=defaults["skip_fbx"],
            person_idx=defaults["person_idx"],
            smooth_cutoff=defaults["smooth"],
            ground_alignment_mode=defaults["ground_alignment_mode"],
            vertical_translation_mode=defaults["vertical_translation_mode"],
            post_ik_foot_snap_mode=defaults["post_ik_foot_snap_mode"],
            ik_backend=defaults["ik_backend"],
            save_graph=defaults["save_graph"],
        )
        return

    from run_pipeline import run_pipeline

    input_path = _require_path(
        input_override or defaults["input_video_path"],
        "Input video not provided. Set --input or input.video_path in Config.toml.",
    )
    video_inputs = collect_video_inputs(input_path)
    batch_mode = is_batch_input(input_path)
    configured_output_dir = output_override or defaults["output_dir"]
    for video_input in video_inputs:
        resolved_output_dir = str(
            resolve_video_output_dir(
                video_path=video_input,
                configured_output_dir=configured_output_dir,
                batch_mode=batch_mode,
            )
        )
        if batch_mode:
            print(f"\n[BATCH] Processing {video_input.name}")
        run_pipeline(
            input_path=str(video_input),
            subject_height=defaults["height"],
            subject_mass=defaults["mass"],
            output_dir=resolved_output_dir,
            target_fps=defaults["fps"],
            device=defaults["device"],
            config_path=config_path,
            skip_ik=defaults["skip_ik"],
            visualize=False,
            save_fbx=defaults["save_fbx"],
            global_translation=defaults["global_translation"],
            detector=defaults["detector"],
            segmentor=defaults["segmentor"],
            fov=defaults["fov"],
            use_mask=defaults["use_mask"],
            smooth_cutoff=defaults["smooth"],
            ground_alignment_mode=defaults["ground_alignment_mode"],
            vertical_translation_mode=defaults["vertical_translation_mode"],
            single_person=defaults["single_person"],
            support_surface_mode=defaults["support_surface_mode"],
            post_ik_foot_snap_mode=defaults["post_ik_foot_snap_mode"],
            ik_backend=defaults["ik_backend"],
            save_graph=defaults["save_graph"],
            save_mesh_video=defaults["save_mesh_video"],
            save_mesh_sequence=defaults["save_mesh_sequence"],
            mesh_sequence_format=defaults["mesh_sequence_format"],
        )


if __name__ == "__main__":
    main()
