"""Shared root-CLI option helpers backed by repository config defaults."""

from __future__ import annotations

import argparse
from typing import Any, Optional

from utils.cli_utils import str_to_bool
from utils.io_utils import load_config


DETECTOR_CHOICES = ("vitdet", "yolo11", "sam3", "none")
SEGMENTOR_CHOICES = ("sam2", "none")
FOV_CHOICES = ("moge2", "none")
SUPPORT_SURFACE_CHOICES = ("auto", "manual_roi")
GROUND_ALIGNMENT_CHOICES = ("auto", "contact_aware", "per_frame_snap")
VERTICAL_TRANSLATION_CHOICES = ("auto", "legacy_xz_only", "hybrid_support_plane")
POST_IK_FOOT_SNAP_CHOICES = ("off", "auto", "stance_only")
IK_BACKEND_CHOICES = ("direct_opensim", "pose2sim_augmented")
MESH_SEQUENCE_FORMAT_CHOICES = ("ply", "obj")
RUN_MODE_CHOICES = ("full", "inference", "export", "pipeline")


def _display_default(value: Any, *, none_label: str = "none") -> str:
    if value is None:
        return none_label
    return str(value)


def _normalize_component_choice(
    value: Any,
    *,
    allowed: tuple[str, ...],
    fallback: Optional[str],
) -> Optional[str]:
    if value is None:
        return fallback

    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return None if fallback is None else ("none" if "none" in allowed else fallback)
    if normalized in allowed:
        return normalized
    return fallback


def _normalize_choice(
    value: Any,
    *,
    allowed: tuple[str, ...],
    fallback: str,
) -> str:
    normalized = str(value).strip().lower() if value is not None else ""
    return normalized if normalized in allowed else fallback


def _normalize_bool(value: Any, *, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return fallback


def add_boolean_arg(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        flag,
        type=str_to_bool,
        nargs="?",
        const=True,
        default=default,
        help=f"{help_text} (default: {str(bool(default)).lower()})",
    )


def load_cli_defaults(config_path: str | None = None) -> dict[str, Any]:
    """Load repo-local CLI defaults from config, falling back to current behavior."""
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        if config_path is not None:
            raise
        config = {}

    input_cfg = config.get("input", {})
    run_cfg = config.get("run", {})
    subject_cfg = config.get("subject", {})
    sam3d_cfg = config.get("sam3d", {})
    processing_cfg = config.get("processing", {})
    opensim_cfg = config.get("opensim", {})
    output_cfg = config.get("output", {})

    raw_ik_backend = opensim_cfg.get("ik_backend")
    raw_mode = run_cfg.get("mode")
    if raw_mode is None:
        mode = "full"
    else:
        mode = str(raw_mode).strip().lower()
        if mode not in RUN_MODE_CHOICES:
            raise ValueError(
                "Unsupported run.mode value in config: "
                f"{raw_mode!r}. Expected one of {RUN_MODE_CHOICES}."
            )
    if raw_ik_backend is None:
        ik_backend = "direct_opensim"
    else:
        ik_backend = str(raw_ik_backend).strip().lower()
        if ik_backend not in IK_BACKEND_CHOICES:
            raise ValueError(
                "Unsupported opensim.ik_backend value in config: "
                f"{raw_ik_backend!r}. Expected one of {IK_BACKEND_CHOICES}."
            )

    legacy_save_mesh_obj = _normalize_bool(
        output_cfg.get("save_mesh_obj"),
        fallback=False,
    )
    raw_mesh_sequence_format = output_cfg.get("mesh_sequence_format")
    if raw_mesh_sequence_format is None and legacy_save_mesh_obj:
        raw_mesh_sequence_format = "obj"
    mesh_sequence_format = _normalize_choice(
        raw_mesh_sequence_format,
        allowed=MESH_SEQUENCE_FORMAT_CHOICES,
        fallback="ply",
    )
    save_fbx = _normalize_bool(
        output_cfg.get("save_fbx"),
        fallback=False,
    )
    skip_fbx = _normalize_bool(
        run_cfg.get("skip_fbx"),
        fallback=not save_fbx,
    )

    return {
        "fps": float(input_cfg.get("fps", 30.0)),
        "input_video_path": input_cfg.get("video_path"),
        "input_video_outputs_path": input_cfg.get("video_outputs_path"),
        "height": float(subject_cfg.get("height", 1.75)),
        "mass": float(subject_cfg.get("mass", 70.0)),
        "mode": mode,
        "device": _normalize_choice(
            sam3d_cfg.get("device"),
            allowed=("cuda", "cpu"),
            fallback="cuda",
        ),
        "detector": _normalize_component_choice(
            sam3d_cfg.get("detector_name"),
            allowed=DETECTOR_CHOICES,
            fallback="vitdet",
        ),
        "segmentor": _normalize_component_choice(
            sam3d_cfg.get("segmentor_name"),
            allowed=SEGMENTOR_CHOICES,
            fallback=None,
        ),
        "fov": _normalize_component_choice(
            sam3d_cfg.get("fov_name"),
            allowed=FOV_CHOICES,
            fallback="moge2",
        ),
        "use_mask": _normalize_bool(
            sam3d_cfg.get("use_mask"),
            fallback=False,
        ),
        "single_person": _normalize_bool(
            processing_cfg.get("single_person"),
            fallback=True,
        ),
        "support_surface_mode": _normalize_choice(
            processing_cfg.get("support_surface_mode"),
            allowed=SUPPORT_SURFACE_CHOICES,
            fallback="auto",
        ),
        "smooth": float(processing_cfg.get("filter_cutoff", 6.0)),
        "ground_alignment_mode": _normalize_choice(
            processing_cfg.get("ground_alignment_mode"),
            allowed=GROUND_ALIGNMENT_CHOICES,
            fallback="auto",
        ),
        "vertical_translation_mode": _normalize_choice(
            processing_cfg.get("vertical_translation_mode"),
            allowed=VERTICAL_TRANSLATION_CHOICES,
            fallback="auto",
        ),
        "post_ik_foot_snap_mode": _normalize_choice(
            processing_cfg.get("post_ik_foot_snap_mode"),
            allowed=POST_IK_FOOT_SNAP_CHOICES,
            fallback="off",
        ),
        "ik_backend": ik_backend,
        "output_dir": output_cfg.get("directory"),
        "skip_inference": _normalize_bool(
            run_cfg.get("skip_inference"),
            fallback=False,
        ),
        "skip_ik": _normalize_bool(
            run_cfg.get("skip_ik"),
            fallback=False,
        ),
        "skip_fbx": _normalize_bool(
            run_cfg.get("skip_fbx"),
            fallback=skip_fbx,
        ),
        "global_translation": _normalize_bool(
            run_cfg.get("global_translation"),
            fallback=False,
        ),
        "person_idx": int(run_cfg.get("person_index", 0)),
        "save_mesh_video": _normalize_bool(
            output_cfg.get("save_mesh_video"),
            fallback=_normalize_bool(
                output_cfg.get("save_visualization"),
                fallback=False,
            ),
        ),
        "save_mesh_sequence": _normalize_bool(
            output_cfg.get("save_mesh_sequence"),
            fallback=_normalize_bool(
                output_cfg.get("save_mesh_obj"),
                fallback=False,
            ),
        ),
        "save_fbx": save_fbx,
        "save_graph": _normalize_bool(
            output_cfg.get("save_graph"),
            fallback=False,
        ),
        "mesh_sequence_format": mesh_sequence_format,
    }


def load_cli_defaults_from_argv(
    argv: list[str] | None = None,
    *,
    include_config: bool = False,
) -> dict[str, Any]:
    """Load CLI defaults, optionally bootstrapping a user-supplied --config path."""
    if not include_config:
        return load_cli_defaults()

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known_args, _ = bootstrap.parse_known_args(argv)
    return load_cli_defaults(known_args.config)


def add_subject_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--height",
        type=float,
        default=defaults["height"],
        help=f"Subject height (m) (default: {defaults['height']:.2f})",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=defaults["mass"],
        help=f"Subject mass (kg) (default: {defaults['mass']:.1f})",
    )


def add_inference_runtime_args(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
    *,
    include_device: bool,
    include_config: bool,
) -> None:
    parser.add_argument(
        "--fps",
        type=float,
        default=defaults["fps"],
        help=f"Target FPS (default: {defaults['fps']:.1f})",
    )
    if include_device:
        parser.add_argument(
            "--device",
            default=defaults["device"],
            choices=["cuda", "cpu"],
            help=f"Device (default: {defaults['device']})",
        )
    if include_config:
        parser.add_argument("--config", help="Config file path")

    parser.add_argument(
        "--detector",
        default=defaults["detector"],
        choices=DETECTOR_CHOICES,
        help=(
            "Human detector: vitdet, yolo11, sam3, or none "
            f"(default: {_display_default(defaults['detector'])})"
        ),
    )
    parser.add_argument(
        "--segmentor",
        default=defaults["segmentor"],
        choices=SEGMENTOR_CHOICES,
        help=(
            "Segmentor: sam2 or none "
            f"(default: {_display_default(defaults['segmentor'])})"
        ),
    )
    parser.add_argument(
        "--fov",
        default=defaults["fov"],
        choices=FOV_CHOICES,
        help=(
            "FOV estimator: moge2 or none "
            f"(default: {_display_default(defaults['fov'])})"
        ),
    )
    add_boolean_arg(
        parser,
        "--use-mask",
        default=defaults["use_mask"],
        help_text="Use segmentation mask (requires segmentor)",
    )
    parser.add_argument(
        "--single_person",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=defaults["single_person"],
        help=(
            "Prompt once to choose a single tracked person "
            f"(default: {str(bool(defaults['single_person'])).lower()})"
        ),
    )
    parser.add_argument(
        "--support-surface-mode",
        choices=SUPPORT_SURFACE_CHOICES,
        default=defaults["support_surface_mode"],
        help=(
            "Support-surface selection mode "
            f"(default: {_display_default(defaults['support_surface_mode'], none_label='auto')})"
        ),
    )
    parser.add_argument(
        "--save-mesh-video",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=defaults["save_mesh_video"],
        help=(
            "Save Stage 1 mesh overlay video under mesh_vis/overlay.mp4 "
            f"(default: {str(bool(defaults['save_mesh_video'])).lower()})"
        ),
    )
    parser.add_argument(
        "--save-mesh-sequence",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=defaults["save_mesh_sequence"],
        help=(
            "Save per-frame mesh files for DCC import under mesh_export/ "
            f"(default: {str(bool(defaults['save_mesh_sequence'])).lower()})"
        ),
    )
    parser.add_argument(
        "--mesh-sequence-format",
        choices=MESH_SEQUENCE_FORMAT_CHOICES,
        default=defaults["mesh_sequence_format"],
        help=(
            "Mesh export format for --save-mesh-sequence "
            f"(default: {defaults['mesh_sequence_format']})"
        ),
    )


def add_processing_args(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
    parser.add_argument(
        "--ik-backend",
        choices=IK_BACKEND_CHOICES,
        default=defaults["ik_backend"],
        help=(
            "IK backend selection "
            f"(default: {defaults['ik_backend']})"
        ),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=defaults["smooth"],
        help=(
            "Smoothing cutoff frequency in Hz (0 to disable, "
            f"default: {defaults['smooth']:.1f})"
        ),
    )
    parser.add_argument(
        "--ground-alignment-mode",
        choices=GROUND_ALIGNMENT_CHOICES,
        default=defaults["ground_alignment_mode"],
        help=(
            "Ground alignment strategy "
            f"(default: {defaults['ground_alignment_mode']})"
        ),
    )
    parser.add_argument(
        "--vertical-translation-mode",
        choices=VERTICAL_TRANSLATION_CHOICES,
        default=defaults["vertical_translation_mode"],
        help=(
            "Vertical translation strategy "
            f"(default: {defaults['vertical_translation_mode']})"
        ),
    )
    parser.add_argument(
        "--post-ik-foot-snap",
        choices=POST_IK_FOOT_SNAP_CHOICES,
        default=defaults["post_ik_foot_snap_mode"],
        help=(
            "Postprocess MOT after IK to reduce stance-phase foot hover "
            f"(default: {defaults['post_ik_foot_snap_mode']})"
        ),
    )
    add_boolean_arg(
        parser,
        "--save_graph",
        default=defaults["save_graph"],
        help_text="Save TRC coordinate and MOT angle graphs under graphs/",
    )
