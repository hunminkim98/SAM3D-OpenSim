"""Shared Config.toml loading and normalization helpers."""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


_PATH_FIELDS = (
    ("input", "video_path"),
    ("input", "video_outputs_path"),
    ("sam3d", "sam3d_root"),
    ("sam3d", "checkpoint"),
    ("sam3d", "mhr_path"),
    ("sam3d", "detector_path"),
    ("sam3d", "segmentor_path"),
    ("sam3d", "fov_path"),
    ("opensim", "model"),
    ("opensim", "markers_xml"),
    ("output", "directory"),
)

_EMPTY_TO_NONE_FIELDS = set(_PATH_FIELDS)
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def source_checkout_root() -> Path | None:
    candidate = _repo_root()
    required_paths = (
        candidate / "run_inference.py",
        candidate / "run_export.py",
        candidate / "run_full_pipeline.py",
        candidate / "src",
        candidate / "utils",
        candidate / "config",
    )
    if all(path.exists() for path in required_paths):
        return candidate
    return None


def project_root() -> Path:
    return source_checkout_root() or _repo_root()


def runtime_workspace() -> Path:
    return Path.cwd()


def default_config_path() -> Path:
    repo_root = source_checkout_root()
    if repo_root is not None:
        repo_default = repo_root / "Config.toml"
        if repo_default.exists():
            return repo_default
    return Path(resources.files("sam3d_opensim").joinpath("data/default_config.toml"))


def legacy_config_path() -> Path:
    return _repo_root() / "config" / "config.yaml"


def default_marker_mapping_path() -> Path:
    return _repo_root() / "config" / "marker_mapping.yaml"


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".toml":
        if tomllib is None:  # pragma: no cover
            raise RuntimeError("TOML parsing requires Python 3.11+")
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_known_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = _deep_merge({}, config)
    for section, key in _PATH_FIELDS:
        section_data = resolved.get(section)
        if not isinstance(section_data, dict):
            continue
        value = section_data.get(key)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            section_data[key] = ""
            continue
        if _WINDOWS_PATH_RE.match(stripped):
            if os.name == "nt":
                candidate = Path(stripped).expanduser()
            else:
                pure = PureWindowsPath(stripped)
                drive = pure.drive.rstrip(":").lower()
                candidate = Path("/mnt") / drive / Path(*pure.parts[1:])
        else:
            candidate = Path(stripped).expanduser()
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
        section_data[key] = str(candidate)
    return resolved


def _normalize_optional_values(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _deep_merge({}, config)
    for section, key in _EMPTY_TO_NONE_FIELDS:
        section_data = normalized.get(section)
        if not isinstance(section_data, dict):
            continue
        value = section_data.get(key)
        if isinstance(value, str) and not value.strip():
            section_data[key] = None

    sam3d_cfg = normalized.setdefault("sam3d", {})
    for key in ("detector_name", "segmentor_name", "fov_name"):
        value = sam3d_cfg.get(key)
        if isinstance(value, str) and not value.strip():
            sam3d_cfg[key] = None

    output_cfg = normalized.setdefault("output", {})
    if isinstance(output_cfg.get("mesh_sequence_format"), str):
        output_cfg["mesh_sequence_format"] = output_cfg["mesh_sequence_format"].lower()

    return normalized


def load_pipeline_config(config_path: str | None = None) -> dict[str, Any]:
    default_path = default_config_path()
    defaults = _resolve_known_paths(_read_config_file(default_path), default_path.parent)

    if config_path is None:
        return _normalize_optional_values(defaults)

    candidate = Path(config_path).expanduser()
    override = _resolve_known_paths(_read_config_file(candidate), candidate.parent)
    return _normalize_optional_values(_deep_merge(defaults, override))
