"""Windows-oriented tool path discovery helpers."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Iterable, Sequence


_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _iter_path_forms(candidate: str | Path) -> Iterable[Path]:
    """Yield the candidate as-is and, on WSL, as a translated /mnt path."""
    text = str(candidate).strip().strip('"')
    if not text:
        return

    direct = Path(text).expanduser()
    yield direct

    translated = _translate_windows_path(text)
    if translated and translated != direct:
        yield translated


def _translate_windows_path(candidate: str) -> Path | None:
    """Translate C:\\... paths to /mnt/c/... when running under WSL/Linux."""
    if os.name == "nt" or not _WINDOWS_PATH_RE.match(candidate):
        return None

    pure = PureWindowsPath(candidate)
    drive = pure.drive.rstrip(":").lower()
    if not drive:
        return None

    return Path("/mnt") / drive / Path(*pure.parts[1:])


def _first_existing_path(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        for path in _iter_path_forms(candidate):
            if path.exists():
                return path
    return None


def _current_conda_root() -> Path | None:
    """Infer the active Conda root from the current Python or Conda env vars."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_path = _first_existing_path([conda_prefix])
        if prefix_path:
            if prefix_path.parent.name.lower() == "envs":
                return prefix_path.parent.parent
            return prefix_path

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        exe_path = _first_existing_path([conda_exe])
        if exe_path:
            return exe_path.parent.parent

    current_python = Path(sys.executable)
    if current_python.parent.parent.name.lower() == "envs":
        return current_python.parent.parent.parent
    if current_python.parent.name.lower() == "scripts":
        return current_python.parent.parent

    return None


def _candidate_conda_roots() -> list[Path]:
    roots: list[Path] = []

    current_root = _current_conda_root()
    if current_root:
        roots.append(current_root)

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        roots.extend(
            [
                Path(user_profile) / "miniconda3",
                Path(user_profile) / "anaconda3",
            ]
        )

    roots.extend(
        [
            Path(r"C:\ProgramData\miniconda3"),
            Path(r"C:\ProgramData\anaconda3"),
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def resolve_conda_env_python(
    env_name: str,
    override_vars: Sequence[str] = (),
) -> Path | None:
    """Locate a Conda environment's python.exe."""
    override_path = _first_existing_path(
        os.environ[var_name]
        for var_name in override_vars
        if os.environ.get(var_name)
    )
    if override_path:
        return override_path

    candidates = [
        root / "envs" / env_name / "python.exe"
        for root in _candidate_conda_roots()
    ]
    return _first_existing_path(candidates)


def require_conda_env_python(
    env_name: str,
    override_vars: Sequence[str] = (),
) -> Path:
    python_path = resolve_conda_env_python(env_name, override_vars=override_vars)
    if python_path:
        return python_path

    env_var_text = ", ".join(override_vars) if override_vars else "override env vars"
    raise FileNotFoundError(
        f"Could not locate the Python executable for the '{env_name}' Conda environment. "
        f"Set one of [{env_var_text}] or install the environment under your Conda root "
        f"(for example %USERPROFILE%\\miniconda3\\envs\\{env_name})."
    )


def resolve_pose2sim_setup(
    opensim_python: str | Path | None = None,
    override_vars: Sequence[str] = (),
) -> Path | None:
    """Locate Pose2Sim's OpenSim setup asset directory."""
    override_path = _first_existing_path(
        os.environ[var_name]
        for var_name in override_vars
        if os.environ.get(var_name)
    )
    if override_path:
        return override_path

    candidates: list[Path] = []
    if opensim_python:
        env_root = Path(opensim_python).parent
        candidates.extend(
            [
                env_root / "Lib" / "site-packages" / "pose2sim" / "OpenSim_Setup",
                env_root / "lib" / "site-packages" / "pose2sim" / "OpenSim_Setup",
            ]
        )

    return _first_existing_path(candidates)


def require_pose2sim_setup(
    opensim_python: str | Path | None = None,
    override_vars: Sequence[str] = (),
) -> Path:
    setup_dir = resolve_pose2sim_setup(opensim_python=opensim_python, override_vars=override_vars)
    if setup_dir:
        return setup_dir

    env_var_text = ", ".join(override_vars) if override_vars else "override env vars"
    raise FileNotFoundError(
        "Could not locate Pose2Sim's OpenSim_Setup directory. "
        f"Set one of [{env_var_text}] or verify that Pose2Sim is installed in the Pose2Sim environment."
    )


def _blender_version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)*)", path.name)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))


def resolve_blender_executable(override_vars: Sequence[str] = ()) -> Path | None:
    """Locate Blender's executable from overrides or common install locations."""
    override_path = _first_existing_path(
        os.environ[var_name]
        for var_name in override_vars
        if os.environ.get(var_name)
    )
    if override_path:
        return override_path

    base_dirs = [
        Path(base_dir) / "Blender Foundation"
        for var_name in ("PROGRAMFILES", "PROGRAMW6432", "PROGRAMFILES(X86)")
        for base_dir in [os.environ.get(var_name)]
        if base_dir
    ]
    base_dirs.extend(
        [
            Path(r"C:\Program Files\Blender Foundation"),
            Path(r"C:\Program Files (x86)\Blender Foundation"),
        ]
    )

    candidates: list[Path] = []
    for base_dir in base_dirs:
        translated_base = _first_existing_path([base_dir])
        if not translated_base:
            continue
        version_dirs = sorted(
            [path for path in translated_base.glob("Blender *") if path.is_dir()],
            key=_blender_version_key,
            reverse=True,
        )
        candidates.extend(version_dir / "blender.exe" for version_dir in version_dirs)

    return _first_existing_path(candidates)
