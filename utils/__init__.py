"""Utility modules for SAM3D Body to OpenSim pipeline."""

from importlib import import_module


__all__ = [
    "extract_frames",
    "get_video_info",
    "load_config",
    "load_marker_mapping",
    "save_json",
    "load_json",
]


def __getattr__(name):
    """Load utility helpers lazily so lightweight submodules stay import-safe."""
    if name in {"extract_frames", "get_video_info"}:
        module = import_module(".video_utils", __name__)
        return getattr(module, name)

    if name in {"load_config", "load_marker_mapping", "save_json", "load_json"}:
        module = import_module(".io_utils", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
