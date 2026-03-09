#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export OpenSim .mot file to FBX using the metarig_skely skeleton.

Uses the expert-verified rotation mappings for the Endorfina metarig skeleton,
which has arms-down rest pose and standard bone naming (spine, thigh.R, etc.).

The skeleton template is loaded from Import_OS4_Patreon_Aitor_Skely.blend.

Rotation mappings (all in Euler XYZ, values converted to radians):
- spine (pelvis):     (-pelvis_tilt, pelvis_rotation, pelvis_list)
- spine location:     (-pelvis_tz, pelvis_ty, pelvis_tx)
- thigh.R:            (-hip_flexion_r, -hip_rotation_r, -hip_adduction_r)
- shin.R:             (+knee_angle_r, 0, 0)
- foot.R:             (-ankle_angle_r, 0, 0)
- thigh.L:            (-hip_flexion_l, +hip_rotation_l, +hip_adduction_l)
- shin.L:             (+knee_angle_l, 0, 0)
- foot.L:             (-ankle_angle_l, 0, 0)
- spine.001 (lumbar):  (-lumbar_extension, lumbar_rotation, lumbar_bending)
- spine.002 (thorax):  (-thorax_extension, thorax_rotation, thorax_bending)
- upper_arm.R:         (-arm_flex_r, -arm_rot_r, -arm_add_r)
- forearm.R:           (-elbow_flex_r, -pro_sup_r, 0)  # Added forearm pronation/supination
- upper_arm.L:         (-arm_flex_l, +arm_rot_l, +arm_add_l)
- forearm.L:           (-elbow_flex_l, +pro_sup_l, 0)  # Added forearm pronation/supination

Usage:
    blender --background Import_OS4_Patreon_Aitor_Skely.blend --python export_fbx_skely.py -- \\
        --mot motion.mot --output motion.fbx [--fps 30]
"""

import os
import sys
from math import radians

# Parse arguments after '--'
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser(description='Export OpenSim .mot to FBX')
parser.add_argument('--mot', required=True, help='Path to .mot motion file')
parser.add_argument('--output', '-o', required=True, help='Output FBX file path')
parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
parser.add_argument('--rig', default='metarig_skely', help='Armature name (default: metarig_skely)')
args = parser.parse_args(argv)

import bpy
import numpy as np


def unwrap_angle(angles):
    """Unwrap angle array to avoid discontinuities at +/-180 degrees."""
    unwrapped = angles.copy()
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]
        if diff > 180:
            unwrapped[i:] -= 360
        elif diff < -180:
            unwrapped[i:] += 360
    return unwrapped


def preprocess_mot_data(headers, data):
    """
    Minimal preprocessing - only unwrap rotation to avoid ±180° discontinuities.
    The MOT data from OpenSim IK is already correct, we just need to handle
    the angle wrapping for smooth animation.
    """
    data = np.array(data)

    # Get column indices for rotation angles that might wrap
    rot_idx = headers.index('pelvis_rotation') if 'pelvis_rotation' in headers else None

    if rot_idx is not None:
        # Only unwrap to avoid discontinuities - don't normalize or smooth
        rotations = data[:, rot_idx].copy()
        unwrapped = unwrap_angle(rotations)
        data[:, rot_idx] = unwrapped
        print(f"  Pelvis rotation: unwrapped (range: {unwrapped.min():.1f}° to {unwrapped.max():.1f}°)")

    return data.tolist()


def read_mot_file(mot_path):
    """Read OpenSim .mot file, return (headers, data_rows, in_degrees)."""
    with open(mot_path, 'r') as f:
        lines = f.readlines()

    in_degrees = True
    data_start = 0
    for i, line in enumerate(lines):
        if 'inDegrees=yes' in line:
            in_degrees = True
        elif 'inDegrees=no' in line:
            in_degrees = False
        if line.strip().startswith('endheader'):
            data_start = i + 1
            break

    headers = None
    data = []
    for i, line in enumerate(lines[data_start:]):
        parts = line.strip().split('\t')
        if i == 0:
            headers = parts
        else:
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                continue

    return headers, data, in_degrees


def get_val(row, headers, col_name, default=0.0):
    """Get value from data row by column name."""
    if col_name in headers:
        idx = headers.index(col_name)
        if idx < len(row):
            return row[idx]
    return default


def apply_motion(rig_name, mot_path, target_fps=30):
    """Apply .mot motion data to the skeleton using expert-verified mappings."""
    headers, data, in_degrees = read_mot_file(mot_path)
    print(f"Loaded {mot_path}: {len(data)} frames, {len(headers)} columns, degrees={in_degrees}")

    # Preprocess to fix rotation discontinuities
    print("Preprocessing motion data...")
    data = preprocess_mot_data(headers, data)

    rig = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='POSE')

    # Calculate frame skip for FPS conversion
    if len(data) > 1:
        t_ini = data[0][0]
        t_end = data[-1][0]
        source_fps = (len(data) - 1) / (t_end - t_ini) if t_end > t_ini else 30
        fps_skip = max(1, int(source_fps / target_fps))
        print(f"  Source FPS: {source_fps:.1f}, target: {target_fps}, skip: {fps_skip}")
    else:
        fps_skip = 1

    frame = 0
    for i in range(0, len(data), fps_skip):
        row = data[i]

        # Read all DOFs
        pelvis_tx = get_val(row, headers, 'pelvis_tx', 0)
        pelvis_ty = get_val(row, headers, 'pelvis_ty', 0)
        pelvis_tz = get_val(row, headers, 'pelvis_tz', 0)
        pelvis_tilt = get_val(row, headers, 'pelvis_tilt', 0)
        pelvis_list = get_val(row, headers, 'pelvis_list', 0)
        pelvis_rotation = get_val(row, headers, 'pelvis_rotation', 0)

        hip_flexion_r = get_val(row, headers, 'hip_flexion_r', 0)
        hip_adduction_r = get_val(row, headers, 'hip_adduction_r', 0)
        hip_rotation_r = get_val(row, headers, 'hip_rotation_r', 0)
        knee_angle_r = get_val(row, headers, 'knee_angle_r', 0)
        ankle_angle_r = get_val(row, headers, 'ankle_angle_r', 0)

        hip_flexion_l = get_val(row, headers, 'hip_flexion_l', 0)
        hip_adduction_l = get_val(row, headers, 'hip_adduction_l', 0)
        hip_rotation_l = get_val(row, headers, 'hip_rotation_l', 0)
        knee_angle_l = get_val(row, headers, 'knee_angle_l', 0)
        ankle_angle_l = get_val(row, headers, 'ankle_angle_l', 0)

        # Lumbar - support both naming conventions
        lumbar_ext = get_val(row, headers, 'lumbar_extension',
                    get_val(row, headers, 'L5_S1_Flex_Ext', 0))
        lumbar_bend = get_val(row, headers, 'lumbar_bending',
                     get_val(row, headers, 'L5_S1_Lat_Bending', 0))
        lumbar_rot = get_val(row, headers, 'lumbar_rotation',
                    get_val(row, headers, 'L5_S1_axial_rotation', 0))

        # Thorax - support both naming conventions
        thorax_ext = get_val(row, headers, 'thorax_extension',
                    get_val(row, headers, 'neck_flexion', 0))
        thorax_bend = get_val(row, headers, 'thorax_bending',
                     get_val(row, headers, 'neck_bending', 0))
        thorax_rot = get_val(row, headers, 'thorax_rotation',
                    get_val(row, headers, 'neck_rotation', 0))

        arm_flex_r = get_val(row, headers, 'arm_flex_r', 0)
        arm_add_r = get_val(row, headers, 'arm_add_r', 0)
        arm_rot_r = get_val(row, headers, 'arm_rot_r', 0)
        elbow_flex_r = get_val(row, headers, 'elbow_flex_r', 0)
        pro_sup_r = get_val(row, headers, 'pro_sup_r', 0)  # Forearm pronation/supination

        arm_flex_l = get_val(row, headers, 'arm_flex_l', 0)
        arm_add_l = get_val(row, headers, 'arm_add_l', 0)
        arm_rot_l = get_val(row, headers, 'arm_rot_l', 0)
        elbow_flex_l = get_val(row, headers, 'elbow_flex_l', 0)
        pro_sup_l = get_val(row, headers, 'pro_sup_l', 0)  # Forearm pronation/supination

        # Convert to radians if in degrees
        if in_degrees:
            pelvis_tilt = radians(pelvis_tilt)
            pelvis_list = radians(pelvis_list)
            pelvis_rotation = radians(pelvis_rotation)
            hip_flexion_r = radians(hip_flexion_r)
            hip_adduction_r = radians(hip_adduction_r)
            hip_rotation_r = radians(hip_rotation_r)
            knee_angle_r = radians(knee_angle_r)
            ankle_angle_r = radians(ankle_angle_r)
            hip_flexion_l = radians(hip_flexion_l)
            hip_adduction_l = radians(hip_adduction_l)
            hip_rotation_l = radians(hip_rotation_l)
            knee_angle_l = radians(knee_angle_l)
            ankle_angle_l = radians(ankle_angle_l)
            lumbar_ext = radians(lumbar_ext)
            lumbar_bend = radians(lumbar_bend)
            lumbar_rot = radians(lumbar_rot)
            thorax_ext = radians(thorax_ext)
            thorax_bend = radians(thorax_bend)
            thorax_rot = radians(thorax_rot)
            arm_flex_r = radians(arm_flex_r)
            arm_add_r = radians(arm_add_r)
            arm_rot_r = radians(arm_rot_r)
            elbow_flex_r = radians(elbow_flex_r)
            pro_sup_r = radians(pro_sup_r)
            arm_flex_l = radians(arm_flex_l)
            arm_add_l = radians(arm_add_l)
            arm_rot_l = radians(arm_rot_l)
            elbow_flex_l = radians(elbow_flex_l)
            pro_sup_l = radians(pro_sup_l)

        # === PELVIS (spine bone) ===
        rig.pose.bones['spine'].location.x = -pelvis_tz
        rig.pose.bones['spine'].location.y = pelvis_ty
        rig.pose.bones['spine'].location.z = pelvis_tx
        rig.pose.bones['spine'].keyframe_insert(data_path="location", frame=frame)

        rig.pose.bones['spine'].rotation_euler = np.array([
            -pelvis_tilt, pelvis_rotation, pelvis_list])
        rig.pose.bones['spine'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === RIGHT LEG ===
        rig.pose.bones['thigh.R'].rotation_euler = np.array([
            -hip_flexion_r, -hip_rotation_r, -hip_adduction_r])
        rig.pose.bones['thigh.R'].keyframe_insert(data_path="rotation_euler", frame=frame)

        rig.pose.bones['shin.R'].rotation_euler = np.array([knee_angle_r, 0, 0])
        rig.pose.bones['shin.R'].keyframe_insert(data_path="rotation_euler", frame=frame)

        rig.pose.bones['foot.R'].rotation_euler = np.array([-ankle_angle_r, 0, 0])
        rig.pose.bones['foot.R'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === LEFT LEG ===
        rig.pose.bones['thigh.L'].rotation_euler = np.array([
            -hip_flexion_l, hip_rotation_l, hip_adduction_l])
        rig.pose.bones['thigh.L'].keyframe_insert(data_path="rotation_euler", frame=frame)

        rig.pose.bones['shin.L'].rotation_euler = np.array([knee_angle_l, 0, 0])
        rig.pose.bones['shin.L'].keyframe_insert(data_path="rotation_euler", frame=frame)

        rig.pose.bones['foot.L'].rotation_euler = np.array([-ankle_angle_l, 0, 0])
        rig.pose.bones['foot.L'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === LUMBAR (spine.001) ===
        rig.pose.bones['spine.001'].rotation_euler = np.array([
            -lumbar_ext, lumbar_rot, lumbar_bend])
        rig.pose.bones['spine.001'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === THORAX (spine.002) ===
        rig.pose.bones['spine.002'].rotation_euler = np.array([
            -thorax_ext, thorax_rot, thorax_bend])
        rig.pose.bones['spine.002'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === RIGHT ARM ===
        rig.pose.bones['upper_arm.R'].rotation_euler = np.array([
            -arm_flex_r, -arm_rot_r, -arm_add_r])
        rig.pose.bones['upper_arm.R'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # Forearm with pronation/supination from hand markers
        rig.pose.bones['forearm.R'].rotation_euler = np.array([-elbow_flex_r, -pro_sup_r, 0])
        rig.pose.bones['forearm.R'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # === LEFT ARM ===
        rig.pose.bones['upper_arm.L'].rotation_euler = np.array([
            -arm_flex_l, arm_rot_l, arm_add_l])
        rig.pose.bones['upper_arm.L'].keyframe_insert(data_path="rotation_euler", frame=frame)

        # Forearm with pronation/supination from hand markers
        rig.pose.bones['forearm.L'].rotation_euler = np.array([-elbow_flex_l, pro_sup_l, 0])
        rig.pose.bones['forearm.L'].keyframe_insert(data_path="rotation_euler", frame=frame)

        frame += 1
        if frame % 200 == 0:
            print(f"  Frame {frame}...")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Set scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frame - 1
    bpy.context.scene.render.fps = target_fps

    print(f"  Applied {frame} frames")
    return frame


def cleanup_scene(rig_name):
    """Remove objects not related to the skeleton export."""
    keep_names = set()
    rig = bpy.data.objects.get(rig_name)
    if rig:
        keep_names.add(rig.name)
        # Keep all mesh children of the rig
        for obj in bpy.data.objects:
            if obj.parent == rig and obj.type == 'MESH':
                keep_names.add(obj.name)

    # Remove everything else
    to_remove = []
    for obj in bpy.data.objects:
        if obj.name not in keep_names:
            to_remove.append(obj)

    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    print(f"Scene cleaned: kept {len(keep_names)} objects")


def export_fbx(output_path):
    """Export the scene to FBX."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Select armature and its meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in ['ARMATURE', 'MESH']:
            obj.select_set(True)

    # Log frame range for debugging
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    print(f"  Scene frame range: {frame_start} - {frame_end} ({frame_end - frame_start + 1} frames)")

    # Also check the Action frame range
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            print(f"  Action '{action.name}' range: {action.frame_range[0]:.0f} - {action.frame_range[1]:.0f}")
            # Force the action's manual frame range to match
            action.use_frame_range = True
            action.frame_start = frame_start
            action.frame_end = frame_end

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        use_armature_deform_only=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )

    print(f"Exported FBX: {output_path}")


def main():
    print(f"Motion file: {args.mot}")
    print(f"Output: {args.output}")
    print(f"Rig: {args.rig}")

    # Verify the rig exists
    if args.rig not in bpy.data.objects:
        print(f"ERROR: Armature '{args.rig}' not found!")
        print(f"Available armatures: {[o.name for o in bpy.data.objects if o.type == 'ARMATURE']}")
        return

    # Clean up the scene (remove non-skeleton objects)
    cleanup_scene(args.rig)

    # Apply motion
    apply_motion(args.rig, args.mot, target_fps=args.fps)

    # Export FBX
    export_fbx(args.output)

    print("Done!")


if __name__ == '__main__':
    main()
