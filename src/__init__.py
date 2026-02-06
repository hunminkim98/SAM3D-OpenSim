"""
SAM3D Body to OpenSim Pipeline
==============================

A pipeline for converting SAM3D Body 3D pose estimates to OpenSim motion files.

Modules:
    sam3d_inference: SAM3D Body model wrapper
    keypoint_converter: MHR70 to OpenSim marker conversion
    coordinate_transform: Camera to OpenSim coordinate transformation
    post_processing: Bone normalization and optional smoothing
    trc_exporter: TRC file generation
    opensim_ik: OpenSim inverse kinematics
"""

__version__ = "0.1.0"
__author__ = "SAM3DBodyToOpenSim"

from .sam3d_inference import SAM3DInference
from .keypoint_converter import KeypointConverter
from .coordinate_transform import CoordinateTransformer
from .post_processing import PostProcessor
from .trc_exporter import TRCExporter
from .opensim_ik import OpenSimIK

__all__ = [
    "SAM3DInference",
    "KeypointConverter",
    "CoordinateTransformer",
    "PostProcessor",
    "TRCExporter",
    "OpenSimIK",
]
