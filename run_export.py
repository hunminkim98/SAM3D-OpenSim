#!/usr/bin/env python3
"""
Export SAM3D Body Results to OpenSim
====================================

Reads video_outputs.json and exports to TRC/MOT/FBX.
This is fast - iterate on settings without re-running inference.

Usage:
    # From our inference output
    python run_export.py --input output_dir/video_outputs.json --height 1.69

    # From original SAM3D Body output
    python run_export.py --input C:/Sam3dBody/sam-3d-body/outputs/my_video/video_outputs.json --height 1.75

    # With global translation (for walking videos)
    python run_export.py --input video_outputs.json --height 1.69 --global-translation

    # Skip IK, only generate TRC
    python run_export.py --input video_outputs.json --height 1.69 --skip-ik
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from utils.windows_paths import (
    require_conda_env_python,
    require_pose2sim_setup,
    resolve_blender_executable,
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Export SAM3D outputs to OpenSim")
    parser.add_argument("--input", "-i", required=True, help="video_outputs.json file")
    parser.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--fps", type=float, help="Override FPS (default: from metadata)")
    parser.add_argument("--global-translation", action="store_true", help="Track global movement")
    parser.add_argument("--skip-ik", action="store_true", help="Skip OpenSim IK")
    parser.add_argument("--skip-fbx", action="store_true", help="Skip FBX export")
    parser.add_argument("--person", type=int, default=0, help="Person index if multiple detected")
    parser.add_argument("--smooth", type=float, default=6.0, help="Smoothing cutoff frequency in Hz (0 to disable)")
    parser.add_argument(
        "--ground-alignment-mode",
        choices=["auto", "contact_aware", "per_frame_snap"],
        default="auto",
        help="Ground alignment strategy (default: auto)",
    )
    parser.add_argument(
        "--vertical-translation-mode",
        choices=["auto", "legacy_xz_only", "hybrid_support_plane"],
        default="auto",
        help="Vertical translation strategy when global translation is enabled (default: auto)",
    )
    parser.add_argument(
        "--post-ik-foot-snap",
        choices=["off", "auto", "stance_only"],
        default="off",
        help="Postprocess MOT after IK to reduce stance-phase foot hover (default: off)",
    )
    return parser.parse_args()


def load_sam3d_outputs(json_path: str) -> dict:
    """Load SAM3D Body outputs from JSON file."""
    json_path = Path(json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check format - list of frames with outputs
    if not isinstance(data, list):
        raise ValueError("Expected list of frame outputs")

    return data


def extract_keypoints_and_cam(data: list, person_idx: int = 0) -> tuple:
    """
    Extract keypoints and camera params from SAM3D JSON data.

    Returns:
        keypoints_3d: (N, 70, 3) array
        cam_translations: (N, 3) array
        focal_lengths: (N,) array
        valid_frames: (N,) bool array
    """
    num_frames = len(data)
    keypoints_3d = np.zeros((num_frames, 70, 3), dtype=np.float32)
    cam_translations = np.zeros((num_frames, 3), dtype=np.float32)
    focal_lengths = np.zeros(num_frames, dtype=np.float32)
    valid_frames = np.zeros(num_frames, dtype=bool)

    for i, frame_data in enumerate(data):
        outputs = frame_data.get("outputs", [])
        if len(outputs) > person_idx:
            person = outputs[person_idx]
            kp3d = person.get("pred_keypoints_3d", [])
            if len(kp3d) == 70:
                keypoints_3d[i] = np.array(kp3d)
                valid_frames[i] = True

            cam_t = person.get("pred_cam_t", [0, 0, 5])
            cam_translations[i] = np.array(cam_t)

            focal_lengths[i] = person.get("focal_length", 1000.0)

    return keypoints_3d, cam_translations, focal_lengths, valid_frames


def run_export(
    json_path: str,
    output_dir: str,
    subject_height: float,
    subject_mass: float,
    fps: float,
    global_translation: bool,
    skip_ik: bool,
    skip_fbx: bool,
    person_idx: int,
    smooth_cutoff: float = 6.0,
    ground_alignment_mode: str = "auto",
    vertical_translation_mode: str = "auto",
    post_ik_foot_snap_mode: str = "off",
):
    """Export SAM3D outputs to TRC/MOT/FBX."""
    start_time = time.time()

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAM3D Body Export to OpenSim")
    print(f"{'='*60}")
    print(f"Input: {json_path}")
    print(f"Output: {output_dir}")
    print(f"Subject: height={subject_height}m, mass={subject_mass}kg")
    print(f"Global translation: {'ENABLED' if global_translation else 'disabled'}")
    print(f"Ground alignment: {ground_alignment_mode}")
    print(f"Vertical translation: {vertical_translation_mode}")
    print(f"Post-IK foot snap: {post_ik_foot_snap_mode}")
    print(f"{'='*60}\n")

    # Load data
    print("[1/5] Loading SAM3D outputs...")
    data = load_sam3d_outputs(json_path)
    keypoints_3d, cam_translations, focal_lengths, valid_frames = extract_keypoints_and_cam(data, person_idx)
    print(f"  Loaded {len(data)} frames, {np.sum(valid_frames)} valid detections")

    # Try to get FPS and clip-level scene-ground metadata from metadata
    meta_path = json_path.parent / "inference_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    if fps is None:
        if meta:
            fps = meta.get("fps", 30.0)
            print(f"  FPS from metadata: {fps}")
        else:
            fps = 30.0
            print(f"  Using default FPS: {fps}")

    # Post-process keypoints
    print("\n[2/5] Post-processing keypoints...")
    from src.post_processing import PostProcessor
    from src.coordinate_transform import CoordinateTransformer
    from src.moge_scene_ground import extract_scene_ground_arrays_from_json
    from src.post_ik_foot_snap import build_post_ik_contact_meta

    use_smoothing = smooth_cutoff > 0
    if use_smoothing:
        print(f"  Smoothing enabled: {smooth_cutoff} Hz cutoff")
    post_processor = PostProcessor(smooth_filter=use_smoothing, filter_cutoff=smooth_cutoff, normalize_bones=True)
    keypoints_processed = post_processor.process(keypoints_3d, fps=fps, subject_height=subject_height)
    scene_ground_data = extract_scene_ground_arrays_from_json(data, person_idx=person_idx)
    clip_scene_ground = meta.get("scene_ground", {}) if meta else {}
    if clip_scene_ground:
        scene_ground_data.update(
            {
                "normal_cam": clip_scene_ground.get("normal_cam"),
                "offset_cam": clip_scene_ground.get("offset_cam"),
                "clip_plane_confidence": clip_scene_ground.get("confidence"),
                "clip_plane_inlier_ratio": clip_scene_ground.get("inlier_ratio"),
                "support_surface_mode_applied": clip_scene_ground.get("support_surface_mode_applied"),
                "support_surface_selection_status": clip_scene_ground.get("support_surface_selection_status"),
            }
        )
    if scene_ground_data.get("available"):
        print(
            "  Loaded MoGe scene-ground hints "
            f"({int(np.sum(scene_ground_data['valid_frames']))} frames)"
        )

    # Transform coordinates
    transformer = CoordinateTransformer(subject_height=subject_height, units="mm")

    if global_translation:
        print("  Applying global translation from cam_t...")

    keypoints_opensim = transformer.transform(
        keypoints_processed,
        camera_translation=cam_translations if global_translation else None,
        center_pelvis=not global_translation,
        align_to_ground=True,
        apply_global_translation=global_translation,
        ground_alignment_mode=ground_alignment_mode,
        scene_ground_data=scene_ground_data,
        vertical_translation_mode=vertical_translation_mode,
    )
    ground_alignment_info = transformer.get_last_ground_alignment_info()
    ground_alignment_message = (
        "  Ground alignment applied: "
        f"{ground_alignment_info.get('applied_mode')} "
        f"(contact_frames={ground_alignment_info.get('contact_frames')}, "
        f"flight_frames={ground_alignment_info.get('flight_frames')})"
    )
    if ground_alignment_info.get("scene_ground_used"):
        ground_alignment_message += (
            f", scene_ground_fused_frames="
            f"{ground_alignment_info.get('scene_ground_fused_frames')}"
        )
    ground_alignment_message += (
        f", vertical_mode={ground_alignment_info.get('vertical_mode')}"
        f", vertical_confident_frames={ground_alignment_info.get('vertical_confident_frames')}"
    )
    if ground_alignment_info.get("manual_plane_anchor_active"):
        ground_alignment_message += (
            ", manual_anchor=on"
            f", manual_bias_l={ground_alignment_info.get('manual_plane_left_bias_m'):.3f}"
            f", manual_bias_r={ground_alignment_info.get('manual_plane_right_bias_m'):.3f}"
        )
    elif ground_alignment_info.get("manual_plane_fallback_reason"):
        ground_alignment_message += (
            f", manual_anchor=off"
            f", manual_reason={ground_alignment_info.get('manual_plane_fallback_reason')}"
        )
    print(ground_alignment_message)
    print("  Coordinate transformation complete")

    post_ik_contact_meta = build_post_ik_contact_meta(
        transformer.get_last_contact_data(),
        ground_alignment_info,
        fps=fps,
    )
    post_ik_contact_meta_path = output_dir / "post_ik_contact_meta.json"
    with open(post_ik_contact_meta_path, "w", encoding="utf-8") as handle:
        json.dump(post_ik_contact_meta, handle, indent=2)
    print(f"  Saved post-IK contact meta: {post_ik_contact_meta_path}")

    # Convert to OpenSim markers
    print("\n[3/5] Converting to OpenSim markers...")
    from src.keypoint_converter import KeypointConverter

    converter = KeypointConverter()
    markers, marker_names = converter.convert(keypoints_opensim, include_derived=True)
    print(f"  Generated {len(marker_names)} markers")

    # Export TRC
    print("\n[4/5] Exporting TRC file...")
    from src.trc_exporter import TRCExporter

    # Get video name from json path or parent folder
    video_name = json_path.parent.name
    if video_name == "." or video_name == "":
        video_name = json_path.stem.replace("video_outputs", "export")

    trc_exporter = TRCExporter(fps=fps, units="mm")
    trc_path = output_dir / f"markers_{video_name}.trc"
    trc_exporter.export(markers, marker_names, str(trc_path))
    print(f"  Saved: {trc_path}")

    results = {"trc": trc_path, "mot": None, "fbx": None}

    # Run IK
    if not skip_ik:
        print("\n[5/5] Running OpenSim IK...")
        mot_path = run_opensim_ik(
            trc_path,
            output_dir,
            subject_height,
            subject_mass,
            post_ik_foot_snap_mode=post_ik_foot_snap_mode,
        )
        results["mot"] = mot_path
    else:
        print("\n[5/5] Skipping OpenSim IK")

    # Export FBX (uses .mot file for joint angles including forearm rotation)
    if not skip_fbx and not skip_ik and mot_path:
        print("\nExporting FBX...")
        fbx_path = run_fbx_export(mot_path, output_dir)
        results["fbx"] = fbx_path

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nOutput files:")
    for name, path in results.items():
        status = "OK" if path and Path(path).exists() else "SKIPPED"
        print(f"  [{status}] {name.upper()}: {path}")
    print(f"{'='*60}\n")

    return results


def run_opensim_ik(
    trc_path: Path,
    output_dir: Path,
    height: float,
    mass: float,
    post_ik_foot_snap_mode: str = "off",
):
    """Run OpenSim IK using Pose2Sim environment."""
    import subprocess
    from src.opensim_marker_spec import (
        build_ik_taskset_xml,
        format_lower_body_marker_summary,
        get_runtime_ik_marker_specs,
    )

    opensim_python = require_conda_env_python(
        "Pose2Sim",
        override_vars=("SAM3D_OPENSIM_OPENSIM_PYTHON", "OPENSIM_PYTHON"),
    )
    pose2sim_setup = require_pose2sim_setup(
        opensim_python=opensim_python,
        override_vars=("SAM3D_OPENSIM_POSE2SIM_SETUP", "POSE2SIM_SETUP"),
    )

    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir).resolve()
    marker_specs = get_runtime_ik_marker_specs(PROJECT_ROOT)
    marker_specs_json = json.dumps(marker_specs)
    marker_task_xml = build_ik_taskset_xml(marker_specs)

    print(f"  IK lower-body markers: {format_lower_body_marker_summary(marker_specs)}")

    # Create IK script
    ik_script = '''
import sys
import json
from pathlib import Path
import opensim as osim

project_root = Path(r"{project_root}")
trc_path = Path(r"{trc_path}")
output_dir = Path(r"{output_dir}")
pose2sim_setup = Path(r"{pose2sim_setup}")
marker_specs = json.loads(r"""{marker_specs_json}""")
marker_task_xml = r"""{marker_task_xml}"""
post_ik_foot_snap_mode = r"{post_ik_foot_snap_mode}"

model_path = pose2sim_setup / "Model_Pose2Sim_simple.osim"

def get_trc_time_range(trc_file):
    with open(trc_file, 'r') as f:
        lines = f.readlines()
    header_values = lines[2].split("\\t")
    data_rate = float(header_values[0])
    num_frames = int(header_values[2])
    return (0.0, (num_frames - 1) / data_rate)

time_range = get_trc_time_range(str(trc_path))
stem = trc_path.stem
mot_path = output_dir / f"{{stem}}_ik.mot"
marker_set_file = output_dir / f"{{stem}}_ik_markers.xml"
setup_file = output_dir / f"{{stem}}_ik_setup.xml"

print(
    "Configured IK foot markers: "
    + ", ".join(
        f"{{spec['name']}}={{spec['weight']:.2f}}"
        for spec in marker_specs
        if spec["name"] in ("LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel")
    )
)

model = osim.Model(str(model_path))
for spec in marker_specs:
    try:
        marker_name = spec["name"]
        body_name = spec["body"]
        x, y, z = spec["location"]
        body = model.getBodySet().get(body_name)
        marker = osim.Marker()
        marker.setName(marker_name)
        marker.setParentFrame(body)
        marker.set_location(osim.Vec3(x, y, z))
        model.addMarker(marker)
    except Exception as e:
        print(f"Warning: {{marker_name}}: {{e}}")

model.finalizeConnections()
model.initSystem()

model_with_markers_path = output_dir / f"{{stem}}_model.osim"
model.printToXML(str(model_with_markers_path))

with open(marker_set_file, 'w') as f:
    f.write(marker_task_xml)

ik_setup_xml = '<?xml version="1.0" encoding="UTF-8" ?>\\n'
ik_setup_xml += '<OpenSimDocument Version="40000">\\n'
ik_setup_xml += '    <InverseKinematicsTool name="ik_tool">\\n'
ik_setup_xml += f'        <model_file>{{str(model_with_markers_path)}}</model_file>\\n'
ik_setup_xml += '        <constraint_weight>20</constraint_weight>\\n'
ik_setup_xml += '        <accuracy>1e-5</accuracy>\\n'
ik_setup_xml += f'        <marker_file>{{str(trc_path)}}</marker_file>\\n'
ik_setup_xml += '        <coordinate_file></coordinate_file>\\n'
ik_setup_xml += f'        <time_range>{{time_range[0]}} {{time_range[1]}}</time_range>\\n'
ik_setup_xml += f'        <output_motion_file>{{str(mot_path)}}</output_motion_file>\\n'
ik_setup_xml += '        <report_errors>true</report_errors>\\n'
ik_setup_xml += '        <report_marker_locations>false</report_marker_locations>\\n'
ik_setup_xml += f'        <results_directory>{{str(output_dir)}}</results_directory>\\n'
ik_setup_xml += f'        <IKTaskSet file="{{str(marker_set_file)}}"/>\\n'
ik_setup_xml += '    </InverseKinematicsTool>\\n'
ik_setup_xml += '</OpenSimDocument>\\n'

with open(setup_file, 'w') as f:
    f.write(ik_setup_xml)

try:
    ik_tool = osim.InverseKinematicsTool(str(setup_file))
    ik_tool.run()
    print(f"SUCCESS: {{mot_path}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    raise

if post_ik_foot_snap_mode != "off":
    sys.path.insert(0, str(project_root))
    from src.post_ik_foot_snap import apply_post_ik_foot_snap

    snap_report = apply_post_ik_foot_snap(
        model_path=model_with_markers_path,
        mot_path=mot_path,
        output_dir=output_dir,
        contact_meta_path=output_dir / "post_ik_contact_meta.json",
        mode=post_ik_foot_snap_mode,
    )
    print(
        "Post-IK foot snap: "
        f"{{snap_report.get('status')}}"
        f", corrected_frames={{snap_report.get('corrected_frames', 0)}}"
        f", max_drop={{snap_report.get('max_applied_drop_m', 0.0):.4f}} m"
    )
'''.format(
        project_root=PROJECT_ROOT,
        trc_path=trc_path,
        output_dir=output_dir,
        pose2sim_setup=pose2sim_setup,
        marker_specs_json=marker_specs_json,
        marker_task_xml=marker_task_xml,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
    )

    ik_script_path = output_dir / "_run_ik.py"
    with open(ik_script_path, 'w') as f:
        f.write(ik_script)

    result = subprocess.run(
        [str(opensim_python), str(ik_script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        # Filter out info messages
        for line in result.stderr.split('\n'):
            if line and not line.startswith('[info]'):
                print(line)

    ik_script_path.unlink(missing_ok=True)

    mot_files = list(output_dir.glob("*_ik.mot"))
    return mot_files[0] if mot_files else None


def run_fbx_export(mot_path: Path, output_dir: Path):
    """Export FBX using Blender with skeleton template.

    Uses the metarig_skely skeleton from Import_OS4_Patreon_Aitor_Skely.blend
    and applies joint angles from the .mot file (including forearm pronation/supination).
    """
    import subprocess

    blend_template = PROJECT_ROOT / "Import_OS4_Patreon_Aitor_Skely.blend"
    blender_script = PROJECT_ROOT / "scripts" / "export_fbx_skely.py"
    blender_path = resolve_blender_executable(
        override_vars=("SAM3D_OPENSIM_BLENDER_PATH", "BLENDER_PATH"),
    )

    mot_path = Path(mot_path).resolve()
    fbx_path = output_dir / f"{mot_path.stem.replace('_ik', '')}.fbx"

    if not blender_path:
        print("  Blender not found. Set BLENDER_PATH or install Blender under Program Files.")
        return None

    if not blend_template.exists():
        print(f"  Skeleton template not found: {blend_template}")
        return None

    cmd = [
        str(blender_path), "--background", str(blend_template),
        "--python", str(blender_script),
        "--", "--mot", str(mot_path), "--output", str(fbx_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  FBX: {fbx_path}")
        return fbx_path
    else:
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line and 'Error' in line:
                    print(f"  {line}")
        return None


def main():
    args = parse_args()

    json_path = Path(args.input)
    if not json_path.exists():
        print(f"Error: Input not found: {json_path}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = json_path.parent

    run_export(
        json_path=str(json_path),
        output_dir=str(output_dir),
        subject_height=args.height,
        subject_mass=args.mass,
        fps=args.fps,
        global_translation=args.global_translation,
        skip_ik=args.skip_ik,
        skip_fbx=args.skip_fbx,
        person_idx=args.person,
        smooth_cutoff=args.smooth,
        ground_alignment_mode=args.ground_alignment_mode,
        vertical_translation_mode=args.vertical_translation_mode,
        post_ik_foot_snap_mode=args.post_ik_foot_snap,
    )


if __name__ == "__main__":
    main()
