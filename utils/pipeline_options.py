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


def load_cli_defaults(config_path: str | None = None) -> dict[str, Any]:
    """Load repo-local CLI defaults from config, falling back to current behavior."""
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        config = {}

    input_cfg = config.get("input", {})
    subject_cfg = config.get("subject", {})
    sam3d_cfg = config.get("sam3d", {})
    processing_cfg = config.get("processing", {})

    return {
        "fps": float(input_cfg.get("fps", 30.0)),
        "height": float(subject_cfg.get("height", 1.75)),
        "mass": float(subject_cfg.get("mass", 70.0)),
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
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Use segmentation mask (requires segmentor)",
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


def add_processing_args(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
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
    parser.add_argument(
        "--save_graph",
        action="store_true",
        help="Save TRC coordinate and MOT angle graphs under graphs/",
    )
