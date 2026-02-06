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
    print(f"{'='*60}\n")

    # Load data
    print("[1/5] Loading SAM3D outputs...")
    data = load_sam3d_outputs(json_path)
    keypoints_3d, cam_translations, focal_lengths, valid_frames = extract_keypoints_and_cam(data, person_idx)
    print(f"  Loaded {len(data)} frames, {np.sum(valid_frames)} valid detections")

    # Try to get FPS from metadata
    meta_path = json_path.parent / "inference_meta.json"
    if fps is None:
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            fps = meta.get("fps", 30.0)
            print(f"  FPS from metadata: {fps}")
        else:
            fps = 30.0
            print(f"  Using default FPS: {fps}")

    # Post-process keypoints
    print("\n[2/5] Post-processing keypoints...")
    from src.post_processing import PostProcessor
    from src.coordinate_transform import CoordinateTransformer

    use_smoothing = smooth_cutoff > 0
    if use_smoothing:
        print(f"  Smoothing enabled: {smooth_cutoff} Hz cutoff")
    post_processor = PostProcessor(smooth_filter=use_smoothing, filter_cutoff=smooth_cutoff, normalize_bones=True)
    keypoints_processed = post_processor.process(keypoints_3d, fps=fps, subject_height=subject_height)

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
    )
    print("  Coordinate transformation complete")

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
        mot_path = run_opensim_ik(trc_path, output_dir, subject_height, subject_mass)
        results["mot"] = mot_path
    else:
        print("\n[5/5] Skipping OpenSim IK")

    # Export FBX
    if not skip_fbx and not skip_ik:
        print("\nExporting FBX...")
        fbx_path = run_fbx_export(trc_path, output_dir)
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


def run_opensim_ik(trc_path: Path, output_dir: Path, height: float, mass: float):
    """Run OpenSim IK using Pose2Sim environment."""
    import subprocess

    OPENSIM_PYTHON = r"C:\ProgramData\anaconda3\envs\Pose2Sim\python.exe"
    pose2sim_setup = Path(r"C:\ProgramData\anaconda3\envs\Pose2Sim\Lib\site-packages\pose2sim\OpenSim_Setup")

    trc_path = Path(trc_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Create IK script
    ik_script = '''
import sys
from pathlib import Path
import opensim as osim

trc_path = Path(r"{trc_path}")
output_dir = Path(r"{output_dir}")
pose2sim_setup = Path(r"{pose2sim_setup}")

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

MARKER_WEIGHTS = [
    ("Nose", 0.8), ("LEye", 0.4), ("REye", 0.4), ("LEar", 0.6), ("REar", 0.6),
    ("Neck", 1.0), ("LShoulder", 1.0), ("RShoulder", 1.0),
    ("LElbow", 1.0), ("RElbow", 1.0), ("LWrist", 1.0), ("RWrist", 1.0),
    # Hand markers for forearm rotation
    ("LIndex3", 0.6), ("RIndex3", 0.6),
    ("LMiddleTip", 0.5), ("RMiddleTip", 0.5),
    # Lower body
    ("LHip", 1.0), ("RHip", 1.0), ("LKnee", 1.0), ("RKnee", 1.0),
    ("LAnkle", 1.0), ("RAnkle", 1.0),
]

COCO17_MARKERS = [
    ("Nose", "head", 0.116266, 0.0126096, 0.0),
    ("LEye", "head", 0.08, 0.025, -0.032),
    ("REye", "head", 0.08, 0.025, 0.032),
    ("LEar", "head", -0.02, 0.015, -0.075),
    ("REar", "head", -0.02, 0.015, 0.075),
    ("Neck", "torso", -0.0127516, 0.366307, -0.000509),
    ("LShoulder", "torso", -0.0127516, 0.366307, -0.201574),
    ("RShoulder", "torso", -0.0127516, 0.366307, 0.201574),
    ("LElbow", "humerus_l", 0.025, -0.297955, 0.008738),
    ("RElbow", "humerus_r", 0.025, -0.297955, -0.008738),
    ("LWrist", "radius_l", -0.000174, -0.235096, -0.009744),
    ("RWrist", "radius_r", -0.000174, -0.235096, 0.009744),
    # Hand markers for forearm rotation
    ("LIndex3", "radius_l", 0.02, -0.32, -0.03),
    ("RIndex3", "radius_r", 0.02, -0.32, 0.03),
    ("LMiddleTip", "radius_l", 0.02, -0.42, -0.01),
    ("RMiddleTip", "radius_r", 0.02, -0.42, 0.01),
    # Lower body
    ("LHip", "pelvis", -0.063927, -0.081343, -0.105406),
    ("RHip", "pelvis", -0.063927, -0.081343, 0.105406),
    ("LKnee", "femur_l", -0.005410, -0.386132, -0.005111),
    ("RKnee", "femur_r", -0.005410, -0.386132, 0.005111),
    ("LAnkle", "tibia_l", -0.000286, -0.40805, -0.014960),
    ("RAnkle", "tibia_r", -0.000286, -0.40805, 0.014960),
]

model = osim.Model(str(model_path))
for marker_name, body_name, x, y, z in COCO17_MARKERS:
    try:
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

model_path_out = output_dir / f"{{stem}}_model.osim"
model.printToXML(str(model_path_out))

ik_tool = osim.InverseKinematicsTool()
ik_tool.setModel(model)
ik_tool.setMarkerDataFileName(str(trc_path))
ik_tool.setStartTime(time_range[0])
ik_tool.setEndTime(time_range[1])
ik_tool.setOutputMotionFileName(str(mot_path))
ik_tool.setResultsDir(str(output_dir))

try:
    ik_tool.run()
    print(f"SUCCESS: {{mot_path}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    raise
'''.format(trc_path=trc_path, output_dir=output_dir, pose2sim_setup=pose2sim_setup)

    ik_script_path = output_dir / "_run_ik.py"
    with open(ik_script_path, 'w') as f:
        f.write(ik_script)

    result = subprocess.run(
        [OPENSIM_PYTHON, str(ik_script_path)],
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


def run_fbx_export(trc_path: Path, output_dir: Path):
    """Export FBX using Blender."""
    import subprocess

    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    blender_script = PROJECT_ROOT / "scripts" / "export_fbx_blender.py"

    trc_path = Path(trc_path).resolve()
    fbx_path = output_dir / f"{trc_path.stem}.fbx"

    if not Path(BLENDER_PATH).exists():
        print(f"  Blender not found: {BLENDER_PATH}")
        return None

    cmd = [
        BLENDER_PATH, "--background", "--python", str(blender_script),
        "--", "--trc", str(trc_path), "--output", str(fbx_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  FBX: {fbx_path}")
        return fbx_path
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
    )


if __name__ == "__main__":
    main()
