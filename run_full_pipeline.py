#!/usr/bin/env python3
"""
Full SAM3D Body to OpenSim Pipeline Runner
==========================================

This script runs the complete pipeline:
1. SAM3D Body inference (using sam_3d_body environment)
2. OpenSim IK (using Pose2Sim environment)
3. FBX export (using Blender)

Usage:
    python run_full_pipeline.py --input video.mp4 --height 1.75
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from utils.windows_paths import (
    require_conda_env_python,
    require_pose2sim_setup,
    resolve_blender_executable,
)
from utils.cli_utils import str_to_bool

# Configuration
PROJECT_ROOT = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description="Full SAM3D Body to OpenSim Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--height", type=float, default=1.75, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=70.0, help="Subject mass (kg)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--skip-inference", action="store_true", help="Skip SAM3D inference (use existing)")
    parser.add_argument("--skip-ik", action="store_true", help="Skip OpenSim IK")
    parser.add_argument("--skip-fbx", action="store_true", help="Skip FBX export")
    parser.add_argument("--global-translation", action="store_true", help="Track global movement from cam_t")

    # SAM3D Body component options
    parser.add_argument("--detector", default="vitdet", choices=["vitdet", "yolo11", "sam3", "none"],
                        help="Human detector: vitdet, yolo11, sam3, or none (default: vitdet)")
    parser.add_argument("--segmentor", default=None, choices=["sam2", "none"],
                        help="Segmentor: sam2 or none (default: none)")
    parser.add_argument("--fov", default="moge2", choices=["moge2", "none"],
                        help="FOV estimator: moge2 or none (default: moge2)")
    parser.add_argument("--use-mask", action="store_true",
                        help="Use segmentation mask (requires segmentor)")
    parser.add_argument("--smooth", type=float, default=6.0,
                        help="Smoothing cutoff frequency in Hz (0 to disable, default: 6.0)")
    parser.add_argument(
        "--ground-alignment-mode",
        choices=["auto", "contact_aware", "per_frame_snap"],
        default=None,
        help="Ground alignment strategy (default: from config, usually auto)",
    )
    parser.add_argument(
        "--vertical-translation-mode",
        choices=["auto", "legacy_xz_only", "hybrid_support_plane"],
        default=None,
        help="Vertical translation strategy (default: from config, usually auto)",
    )
    parser.add_argument(
        "--single_person",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Prompt once to choose a single tracked person (default: true)",
    )
    parser.add_argument(
        "--support-surface-mode",
        choices=["auto", "manual_roi"],
        default=None,
        help="Support-surface selection mode (default: from config, usually auto)",
    )
    parser.add_argument(
        "--post-ik-foot-snap",
        choices=["off", "auto", "stance_only"],
        default="off",
        help="Postprocess MOT after IK to reduce stance-phase foot hover (default: off)",
    )

    return parser.parse_args()


def run_sam3d_inference(input_path, output_dir, height, fps, global_translation=False,
                        detector="vitdet", segmentor=None, fov="moge2", use_mask=False,
                        smooth_cutoff=6.0, ground_alignment_mode=None, single_person=True,
                        support_surface_mode=None, vertical_translation_mode=None):
    """Run SAM3D Body inference stage."""
    print("\n" + "=" * 60)
    print("Stage 1: SAM3D Body Inference")
    print("=" * 60)

    sam3d_python = require_conda_env_python(
        "sam_3d_body",
        override_vars=("SAM3D_OPENSIM_SAM3D_PYTHON", "SAM3D_PYTHON"),
    )

    cmd = [
        str(sam3d_python),
        str(PROJECT_ROOT / "run_pipeline.py"),
        "--input", str(input_path),
        "--height", str(height),
        "--output", str(output_dir),
        "--fps", str(fps),
        "--skip-ik",  # We'll do IK separately with the right environment
        "--detector", detector,
        "--fov", fov,
        "--smooth", str(smooth_cutoff),
        "--single_person", str(bool(single_person)).lower(),
    ]

    if ground_alignment_mode:
        cmd.extend(["--ground-alignment-mode", ground_alignment_mode])
    if vertical_translation_mode:
        cmd.extend(["--vertical-translation-mode", vertical_translation_mode])
    if support_surface_mode:
        cmd.extend(["--support-surface-mode", support_surface_mode])

    if segmentor:
        cmd.extend(["--segmentor", segmentor])

    if use_mask:
        cmd.append("--use-mask")

    if global_translation:
        cmd.append("--global-translation")

    print(f"  Detector: {detector}, FOV: {fov}, Segmentor: {segmentor or 'none'}")
    print(f"  Ground alignment: {ground_alignment_mode or 'config/default'}")
    print(f"  Vertical translation: {vertical_translation_mode or 'config/default'}")
    print(f"  Single-person selection: {'ENABLED' if single_person else 'disabled'}")
    print(f"  Support surface: {support_surface_mode or 'config/default'}")
    print(f"  Python: {sam3d_python}")
    if global_translation:
        print("  Global translation: ENABLED")

    print(f"Running: {' '.join(cmd[:3])}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError("SAM3D Body inference failed")

    return True


def run_opensim_ik(output_dir, height, mass, post_ik_foot_snap_mode="off"):
    """Run OpenSim inverse kinematics using MotionBERT-OpenSim approach."""
    print("\n" + "=" * 60)
    print("Stage 2: OpenSim Inverse Kinematics")
    print("=" * 60)

    opensim_python = require_conda_env_python(
        "Pose2Sim",
        override_vars=("SAM3D_OPENSIM_OPENSIM_PYTHON", "OPENSIM_PYTHON"),
    )
    pose2sim_setup = require_pose2sim_setup(
        opensim_python=opensim_python,
        override_vars=("SAM3D_OPENSIM_POSE2SIM_SETUP", "POSE2SIM_SETUP"),
    )
    from src.opensim_marker_spec import (
        build_ik_taskset_xml,
        format_lower_body_marker_summary,
        get_runtime_ik_marker_specs,
    )

    # Convert to absolute path
    output_dir = Path(output_dir).resolve()

    # Find TRC file
    trc_files = list(output_dir.glob("*.trc"))
    if not trc_files:
        raise FileNotFoundError(f"No TRC file found in {output_dir}")
    trc_path = trc_files[0].resolve()
    print(f"TRC file: {trc_path}")
    marker_specs = get_runtime_ik_marker_specs(PROJECT_ROOT)
    marker_specs_json = json.dumps(marker_specs)
    marker_task_xml = build_ik_taskset_xml(marker_specs)
    print(f"IK lower-body markers: {format_lower_body_marker_summary(marker_specs)}")

    # Create IK runner script following MotionBERT-OpenSim approach
    ik_script = """
import opensim as osim
import json
import sys
from pathlib import Path

project_root = Path(r"{project_root}")
trc_path = Path(r"{trc_path}")
output_dir = Path(r"{output_dir}")
pose2sim_setup = Path(r"{pose2sim_setup}")
marker_specs = json.loads(r'''{marker_specs_json}''')
marker_task_xml = r'''{marker_task_xml}'''
post_ik_foot_snap_mode = r"{post_ik_foot_snap_mode}"

model_path = pose2sim_setup / "Model_Pose2Sim_simple.osim"

print(f"Model: {{model_path}}")
print(f"TRC: {{trc_path}}")

def get_trc_time_range(trc_file):
    with open(trc_file, 'r') as f:
        lines = f.readlines()
    header_values = lines[2].split("\\t")
    data_rate = float(header_values[0])
    num_frames = int(header_values[2])
    return (0.0, (num_frames - 1) / data_rate)

time_range = get_trc_time_range(str(trc_path))
print(f"Time range: {{time_range[0]:.3f}} - {{time_range[1]:.3f}} s")

stem = trc_path.stem
mot_path = output_dir / f"{{stem}}_ik.mot"
setup_file = output_dir / f"{{stem}}_ik_setup.xml"
marker_set_file = output_dir / f"{{stem}}_ik_markers.xml"

with open(marker_set_file, 'w') as f:
    f.write(marker_task_xml)

print("Running OpenSim IK...")
print(
    "Configured IK foot markers: "
    + ", ".join(
        f"{{spec['name']}}={{spec['weight']:.2f}}"
        for spec in marker_specs
        if spec["name"] in ("LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel")
    )
)

model = osim.Model(str(model_path))
print("Adding markers to model:")
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
        print(f"  {{marker_name}} -> {{body_name}}")
    except Exception as e:
        print(f"  Warning: Could not add {{marker_name}}: {{e}}")

model.finalizeConnections()
model.initSystem()
print(f"Model initialized with {{model.getMarkerSet().getSize()}} markers")

model_with_markers_path = output_dir / f"{{stem}}_model.osim"
model.printToXML(str(model_with_markers_path))
print(f"Model with markers saved to: {{model_with_markers_path}}")

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
    print(f"SUCCESS: Motion file saved to {{mot_path}}")
except Exception as e:
    print(f"IK Error: {{e}}")
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

print("IK completed successfully!")
""".format(
        project_root=PROJECT_ROOT,
        trc_path=trc_path,
        output_dir=output_dir,
        pose2sim_setup=pose2sim_setup,
        marker_specs_json=marker_specs_json,
        marker_task_xml=marker_task_xml,
        post_ik_foot_snap_mode=post_ik_foot_snap_mode,
    )

    # Write and run script
    ik_script_path = output_dir / "_run_ik.py"
    with open(ik_script_path, 'w') as f:
        f.write(ik_script)

    print(f"Running IK with OpenSim Python...")
    result = subprocess.run(
        [str(opensim_python), str(ik_script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Clean up
    ik_script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print("Warning: IK may have encountered issues")

    # Check if MOT was created
    mot_files = list(output_dir.glob("*.mot"))
    if mot_files:
        print(f"MOT file created: {mot_files[0]}")
        return mot_files[0]
    return None


def run_fbx_export(output_dir):
    """Export to FBX using Blender with skeleton template.

    Uses the metarig_skely skeleton from Import_OS4_Patreon_Aitor_Skely.blend
    and applies joint angles from the .mot file (including forearm pronation/supination).
    """
    print("\n" + "=" * 60)
    print("Stage 3: FBX Export (Blender)")
    print("=" * 60)

    # Convert to absolute path
    output_dir = Path(output_dir).resolve()

    # Find MOT file (joint angles from IK)
    mot_files = list(output_dir.glob("*_ik.mot"))
    if not mot_files:
        raise FileNotFoundError(f"No MOT file found in {output_dir}")
    mot_path = mot_files[0].resolve()

    # Output FBX path
    fbx_path = output_dir / f"{mot_path.stem.replace('_ik', '')}.fbx"

    # Skeleton template and script
    blend_template = PROJECT_ROOT / "Import_OS4_Patreon_Aitor_Skely.blend"
    blender_script = PROJECT_ROOT / "scripts" / "export_fbx_skely.py"
    blender_path = resolve_blender_executable(
        override_vars=("SAM3D_OPENSIM_BLENDER_PATH", "BLENDER_PATH"),
    )

    if not blender_path:
        print("Blender not found. Set BLENDER_PATH or install Blender under Program Files.")
        return None

    if not blend_template.exists():
        print(f"Skeleton template not found: {blend_template}")
        return None

    print(f"MOT file: {mot_path}")
    print(f"Skeleton template: {blend_template.name}")
    print(f"Output FBX: {fbx_path}")
    print(f"Blender: {blender_path}")

    cmd = [
        str(blender_path),
        "--background", str(blend_template),
        "--python", str(blender_script),
        "--",
        "--mot", str(mot_path),
        "--output", str(fbx_path)
    ]

    print("Running Blender...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"SUCCESS: FBX exported to {fbx_path}")
        return fbx_path
    else:
        print(f"Blender output: {result.stdout}")
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line and 'Error' in line:
                    print(f"  {line}")
        return None


def main():
    args = parse_args()

    print("=" * 60)
    print("SAM3D Body to OpenSim - Full Pipeline")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Subject: height={args.height}m, mass={args.mass}kg")
    print(f"FPS: {args.fps}")
    print(f"SAM3D: detector={args.detector}, fov={args.fov}, segmentor={args.segmentor or 'none'}")
    print(f"Ground alignment: {args.ground_alignment_mode or 'config/default'}")
    print(f"Vertical translation: {args.vertical_translation_mode or 'config/default'}")
    print(f"Post-IK foot snap: {args.post_ik_foot_snap}")
    print(f"Single-person selection: {'ENABLED' if args.single_person else 'disabled'}")
    print(f"Support surface: {args.support_surface_mode or 'config/default'}")
    print(f"Global translation: {'ENABLED' if args.global_translation else 'disabled'}")

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"output_{timestamp}_{input_path.stem}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    results = {
        "trc": None,
        "mot": None,
        "fbx": None
    }

    # Stage 1: SAM3D Body inference
    if not args.skip_inference:
        try:
            run_sam3d_inference(
                input_path, output_dir, args.height, args.fps,
                global_translation=args.global_translation,
                detector=args.detector,
                segmentor=args.segmentor,
                fov=args.fov,
                use_mask=args.use_mask,
                smooth_cutoff=args.smooth,
                ground_alignment_mode=args.ground_alignment_mode,
                vertical_translation_mode=args.vertical_translation_mode,
                single_person=args.single_person,
                support_surface_mode=args.support_surface_mode,
            )
            trc_files = list(output_dir.glob("*.trc"))
            if trc_files:
                results["trc"] = trc_files[0]
        except Exception as e:
            print(f"Stage 1 ERROR: {e}")
            sys.exit(1)
    else:
        print("\nSkipping Stage 1 (using existing TRC)")
        trc_files = list(output_dir.glob("*.trc"))
        if trc_files:
            results["trc"] = trc_files[0]

    # Stage 2: OpenSim IK
    if not args.skip_ik:
        try:
            mot_path = run_opensim_ik(
                output_dir,
                args.height,
                args.mass,
                post_ik_foot_snap_mode=args.post_ik_foot_snap,
            )
            results["mot"] = mot_path
        except Exception as e:
            print(f"Stage 2 ERROR: {e}")

    # Stage 3: FBX export
    if not args.skip_fbx:
        try:
            fbx_path = run_fbx_export(output_dir)
            results["fbx"] = fbx_path
        except Exception as e:
            print(f"Stage 3 ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nOutput files:")
    for name, path in results.items():
        status = "OK" if path and Path(path).exists() else "MISSING"
        print(f"  [{status}] {name.upper()}: {path}")

    print(f"\nAll outputs in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
