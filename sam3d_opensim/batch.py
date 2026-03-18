"""Helpers for batch-friendly video input resolution."""

from __future__ import annotations

from pathlib import Path

from utils.io_utils import get_output_dir


VIDEO_FILE_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
}


def is_supported_video_file(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in VIDEO_FILE_EXTENSIONS


def collect_video_inputs(input_path: str | Path) -> list[Path]:
    """Resolve a file path or a directory of videos into concrete video files."""
    candidate = Path(input_path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Input not found: {candidate}")

    if candidate.is_file():
        if not is_supported_video_file(candidate):
            raise ValueError(
                f"Unsupported video file extension: {candidate.suffix or '<none>'}. "
                f"Expected one of {sorted(VIDEO_FILE_EXTENSIONS)}."
            )
        return [candidate]

    files = sorted(
        path
        for path in candidate.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_FILE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported video files found in directory: {candidate}"
        )
    return files


def is_batch_input(input_path: str | Path) -> bool:
    return Path(input_path).expanduser().resolve().is_dir()


def resolve_video_output_dir(
    *,
    video_path: str | Path,
    configured_output_dir: str | Path | None,
    batch_mode: bool,
) -> Path:
    """Resolve per-video output directories for single or batch execution."""
    if configured_output_dir is None:
        return get_output_dir(str(video_path))

    configured_output_dir = Path(configured_output_dir).expanduser().resolve()
    if batch_mode:
        return get_output_dir(str(video_path), base_output_dir=str(configured_output_dir))
    return configured_output_dir
