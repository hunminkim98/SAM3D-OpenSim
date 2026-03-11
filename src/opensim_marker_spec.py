"""
Centralized OpenSim runtime marker/task definitions.

This module keeps the IK marker subset consistent across the direct
`run_export.py` path, the Stage 2 path in `run_full_pipeline.py`, and the
direct OpenSim fallback in `src/opensim_ik.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from utils.io_utils import load_marker_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MarkerLocation = Tuple[float, float, float]
MarkerSpec = Dict[str, object]


FOOT_MARKER_NAMES: Tuple[str, ...] = (
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)

LOWER_BODY_MARKER_NAMES: Tuple[str, ...] = (
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    *FOOT_MARKER_NAMES,
)

DEFAULT_RUNTIME_MARKER_ORDER: Tuple[str, ...] = (
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "Neck",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LIndex3",
    "RIndex3",
    "LMiddleTip",
    "RMiddleTip",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    *FOOT_MARKER_NAMES,
)

# Tuned runtime marker placements already used by the direct IK runners.
_BASE_RUNTIME_MARKERS: Dict[str, Tuple[str, MarkerLocation]] = {
    "Nose": ("head", (0.116266, 0.0126096, 0.0)),
    "LEye": ("head", (0.08, 0.025, -0.032)),
    "REye": ("head", (0.08, 0.025, 0.032)),
    "LEar": ("head", (-0.02, 0.015, -0.075)),
    "REar": ("head", (-0.02, 0.015, 0.075)),
    "Neck": ("torso", (-0.0127516, 0.366307, -0.000509)),
    "LShoulder": ("torso", (-0.0127516, 0.366307, -0.201574)),
    "RShoulder": ("torso", (-0.0127516, 0.366307, 0.201574)),
    "LElbow": ("humerus_l", (0.025, -0.297955, 0.008738)),
    "RElbow": ("humerus_r", (0.025, -0.297955, -0.008738)),
    "LWrist": ("radius_l", (-0.000174, -0.235096, -0.009744)),
    "RWrist": ("radius_r", (-0.000174, -0.235096, 0.009744)),
    "LIndex3": ("radius_l", (0.02, -0.32, -0.03)),
    "RIndex3": ("radius_r", (0.02, -0.32, 0.03)),
    "LMiddleTip": ("radius_l", (0.02, -0.42, -0.01)),
    "RMiddleTip": ("radius_r", (0.02, -0.42, 0.01)),
    "LHip": ("pelvis", (-0.063927, -0.081343, -0.105406)),
    "RHip": ("pelvis", (-0.063927, -0.081343, 0.105406)),
    "LKnee": ("femur_l", (-0.005410, -0.386132, -0.005111)),
    "RKnee": ("femur_r", (-0.005410, -0.386132, 0.005111)),
    "LAnkle": ("tibia_l", (-0.000286, -0.40805, -0.014960)),
    "RAnkle": ("tibia_r", (-0.000286, -0.40805, 0.014960)),
}

_FOOT_MARKER_FALLBACKS: Dict[str, Tuple[str, MarkerLocation]] = {
    "LBigToe": ("toes_l", (0.04, 0.0, 0.02)),
    "LSmallToe": ("toes_l", (0.03, 0.0, -0.03)),
    "LHeel": ("calcn_l", (-0.03, 0.0, 0.0)),
    "RBigToe": ("toes_r", (0.04, 0.0, -0.02)),
    "RSmallToe": ("toes_r", (0.03, 0.0, 0.03)),
    "RHeel": ("calcn_r", (-0.03, 0.0, 0.0)),
}

DEFAULT_RUNTIME_WEIGHT_OVERRIDES: Dict[str, float] = {
    "Nose": 0.8,
    "LEye": 0.4,
    "REye": 0.4,
    "LEar": 0.6,
    "REar": 0.6,
    "Neck": 1.0,
    "LShoulder": 1.0,
    "RShoulder": 1.0,
    "LElbow": 1.0,
    "RElbow": 1.0,
    "LWrist": 1.0,
    "RWrist": 1.0,
    "LIndex3": 0.6,
    "RIndex3": 0.6,
    "LMiddleTip": 0.5,
    "RMiddleTip": 0.5,
    "LHip": 1.0,
    "RHip": 1.0,
    "LKnee": 1.0,
    "RKnee": 1.0,
    "LAnkle": 1.0,
    "RAnkle": 1.0,
}


def _resolve_project_root(project_root: Optional[Path | str]) -> Path:
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root)


def _load_runtime_ik_config(
    project_root: Optional[Path | str] = None,
) -> tuple[Tuple[str, ...], Dict[str, float]]:
    """Load runtime IK subset settings from marker_mapping.yaml when available."""
    root = _resolve_project_root(project_root)
    mapping_path = root / "config" / "marker_mapping.yaml"

    try:
        mapping_config = load_marker_mapping(str(mapping_path))
    except FileNotFoundError:
        mapping_config = {}

    runtime_ik = mapping_config.get("runtime_ik", {})
    raw_order = runtime_ik.get("marker_order", [])
    marker_order: Tuple[str, ...]
    if raw_order:
        marker_order = tuple(str(name) for name in raw_order if str(name).strip())
    else:
        marker_order = DEFAULT_RUNTIME_MARKER_ORDER

    raw_overrides = runtime_ik.get("weight_overrides", {})
    weight_overrides: Dict[str, float] = {}
    for marker_name, value in raw_overrides.items():
        if isinstance(value, (int, float)):
            weight_overrides[str(marker_name)] = float(value)

    if not weight_overrides:
        weight_overrides = DEFAULT_RUNTIME_WEIGHT_OVERRIDES.copy()

    return marker_order, weight_overrides


def _load_marker_xml_locations(markers_xml_path: Path) -> Dict[str, Tuple[str, MarkerLocation]]:
    if not markers_xml_path.exists():
        return {}

    xml_root = ET.parse(markers_xml_path).getroot()
    locations: Dict[str, Tuple[str, MarkerLocation]] = {}

    for marker_el in xml_root.findall(".//Marker"):
        name = marker_el.get("name")
        body = marker_el.findtext("body")
        location_text = marker_el.findtext("location")

        if not name or not body or not location_text:
            continue

        try:
            x_str, y_str, z_str = location_text.split()
            locations[name] = (body, (float(x_str), float(y_str), float(z_str)))
        except ValueError:
            continue

    return locations


def _resolve_marker_body_and_location(
    marker_name: str,
    xml_locations: Dict[str, Tuple[str, MarkerLocation]],
) -> Tuple[str, MarkerLocation]:
    if marker_name in _BASE_RUNTIME_MARKERS:
        return _BASE_RUNTIME_MARKERS[marker_name]

    if marker_name in xml_locations:
        return xml_locations[marker_name]

    if marker_name in _FOOT_MARKER_FALLBACKS:
        return _FOOT_MARKER_FALLBACKS[marker_name]

    raise KeyError(
        f"Runtime IK marker '{marker_name}' is not defined in base markers or marker XML."
    )


def get_runtime_ik_marker_weights(
    project_root: Optional[Path | str] = None,
) -> Dict[str, float]:
    root = _resolve_project_root(project_root)
    weights: Dict[str, float] = {}
    mapping_path = root / "config" / "marker_mapping.yaml"

    try:
        mapping_config = load_marker_mapping(str(mapping_path))
    except FileNotFoundError:
        mapping_config = {}

    raw_weights = mapping_config.get("marker_weights", {})
    for marker_name, value in raw_weights.items():
        if marker_name == "default_finger_weight":
            continue
        if isinstance(value, (int, float)):
            weights[str(marker_name)] = float(value)

    _marker_order, weight_overrides = _load_runtime_ik_config(root)
    weights.update(weight_overrides)
    return weights


def get_runtime_ik_marker_specs(
    project_root: Optional[Path | str] = None,
    markers_xml_path: Optional[Path | str] = None,
) -> List[MarkerSpec]:
    root = _resolve_project_root(project_root)
    runtime_marker_order, _weight_overrides = _load_runtime_ik_config(root)
    weights = get_runtime_ik_marker_weights(root)
    resolved_xml_path = (
        Path(markers_xml_path)
        if markers_xml_path is not None
        else root / "models" / "Markers_MHR70.xml"
    )
    xml_locations = _load_marker_xml_locations(resolved_xml_path)

    specs: List[MarkerSpec] = []
    for marker_name in runtime_marker_order:
        body_name, location = _resolve_marker_body_and_location(
            marker_name,
            xml_locations,
        )

        specs.append(
            {
                "name": marker_name,
                "body": body_name,
                "location": list(location),
                "weight": float(weights.get(marker_name, 0.5)),
            }
        )

    return specs


def build_ik_taskset_xml(
    marker_specs: Sequence[MarkerSpec],
    taskset_name: str = "sam3d_markers",
) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<OpenSimDocument Version="40000">',
        f'    <IKTaskSet name="{taskset_name}">',
        "        <objects>",
    ]
    for spec in marker_specs:
        lines.extend(
            [
                f'            <IKMarkerTask name="{spec["name"]}">',
                "                <apply>true</apply>",
                f'                <weight>{float(spec["weight"]):.6f}</weight>',
                "            </IKMarkerTask>",
            ]
        )
    lines.extend(
        [
            "        </objects>",
            "    </IKTaskSet>",
            "</OpenSimDocument>",
            "",
        ]
    )
    return "\n".join(lines)


def format_marker_weight_summary(marker_specs: Sequence[MarkerSpec]) -> str:
    parts = [
        f'{spec["name"]}={float(spec["weight"]):.2f}'
        for spec in marker_specs
    ]
    return ", ".join(parts)


def format_lower_body_marker_summary(marker_specs: Sequence[MarkerSpec]) -> str:
    lower_body_specs = [
        spec for spec in marker_specs if spec["name"] in LOWER_BODY_MARKER_NAMES
    ]
    return format_marker_weight_summary(lower_body_specs)
