"""
SAM3D Body to OpenSim Pipeline
==============================

Package exports are resolved lazily so lightweight helpers can be imported
without pulling in the full inference stack.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "SAM3DBodyToOpenSim"

__all__ = [
    "SAM3DInference",
    "KeypointConverter",
    "CoordinateTransformer",
    "PostProcessor",
    "TRCExporter",
    "OpenSimIK",
]

_EXPORTS = {
    "SAM3DInference": (".sam3d_inference", "SAM3DInference"),
    "KeypointConverter": (".keypoint_converter", "KeypointConverter"),
    "CoordinateTransformer": (".coordinate_transform", "CoordinateTransformer"),
    "PostProcessor": (".post_processing", "PostProcessor"),
    "TRCExporter": (".trc_exporter", "TRCExporter"),
    "OpenSimIK": (".opensim_ik", "OpenSimIK"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
