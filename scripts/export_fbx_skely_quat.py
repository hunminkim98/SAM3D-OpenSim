#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export OpenSim .mot file to FBX using QUATERNIONS to avoid gimbal lock.

Uses the metarig_skely skeleton template, converting Euler angles to quaternions
for smooth rotation interpolation without gimbal lock issues.
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
parser = argparse.ArgumentParser(description='Export OpenSim .mot to FBX (Quaternion version)')
parser.add_argument('--mot', required=True, help='Path to .mot motion file')
parser.add_argument('--output', '-o', required=True, help='Output FBX file path')
parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
parser.add_argument('--rig', default='metarig_skely', help='Armature name (default: metarig_skely)')
args = parser.parse_args(argv)

import bpy
from mathutils import Euler, Quaternion
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
    """Preprocess motion data - unwrap angles and center translation at origin."""
    data = np.array(data)

    # Unwrap pelvis rotation
    rot_idx = headers.index('pelvis_rotation') if 'pelvis_rotation' in headers else None
    if rot_idx is not None:
        rotations = data[:, rot_idx].copy()
        unwrapped = unwrap_angle(rotations)
        data[:, rot_idx] = unwrapped
        print(f"  Pelvis rotation: unwrapped (range: {unwrapped.min():.1f} to {unwrapped.max():.1f})")

    # Make translation relative to first frame (center at origin)
    tx_idx = headers.index('pelvis_tx') if 'pelvis_tx' in headers else None
    tz_idx = headers.index('pelvis_tz') if 'pelvis_tz' in headers else None

    if tx_idx is not None:
        tx_init = data[0, tx_idx]
        data[:, tx_idx] = data[:, tx_idx] - tx_init
        print(f"  pelvis_tx: centered (was {tx_init:.3f}m, now starts at 0)")

    if tz_idx is not None:
        tz_init = data[0, tz_idx]
        data[:, tz_idx] = data[:, tz_idx] - tz_init
        print(f"  pelvis_tz: centered (was {tz_init:.3f}m, now starts at 0)")

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


def euler_to_quat(x, y, z, order='YZX'):
    """Convert Euler angles (radians) to quaternion using specified order."""
    euler = Euler((x, y, z), order)
    return euler.to_quaternion()


def set_bone_rotation_quat(bone, x, y, z, euler_order='YZX'):
    """Set bone rotation using quaternion (avoids gimbal lock)."""
    quat = euler_to_quat(x, y, z, euler_order)
    bone.rotation_quaternion = quat


def apply_motion(rig_name, mot_path, target_fps=30):
    """Apply .mot motion data to the skeleton using quaternions."""
    headers, data, in_degrees = read_mot_file(mot_path)
    print(f"Loaded {mot_path}: {len(data)} frames, {len(headers)} columns, degrees={in_degrees}")

    # Preprocess
    print("Preprocessing motion data...")
    data = preprocess_mot_data(headers, data)

    rig = bpy.data.objects[rig_name]
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='POSE')

    # Set all bones to QUATERNION rotation mode
    bone_names = ['spine', 'spine.001', 'spine.002',
                  'thigh.R', 'thigh.L', 'shin.R', 'shin.L', 'foot.R', 'foot.L',
                  'upper_arm.R', 'upper_arm.L', 'forearm.R', 'forearm.L']
    for bone_name in bone_names:
        if bone_name in rig.pose.bones:
            rig.pose.bones[bone_name].rotation_mode = 'QUATERNION'
    print("  Set all bones to QUATERNION rotation mode")

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

        lumbar_ext = get_val(row, headers, 'lumbar_extension',
                    get_val(row, headers, 'L5_S1_Flex_Ext', 0))
        lumbar_bend = get_val(row, headers, 'lumbar_bending',
                     get_val(row, headers, 'L5_S1_Lat_Bending', 0))
        lumbar_rot = get_val(row, headers, 'lumbar_rotation',
                    get_val(row, headers, 'L5_S1_axial_rotation', 0))

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
        pro_sup_r = get_val(row, headers, 'pro_sup_r', 0)

        arm_flex_l = get_val(row, headers, 'arm_flex_l', 0)
        arm_add_l = get_val(row, headers, 'arm_add_l', 0)
        arm_rot_l = get_val(row, headers, 'arm_rot_l', 0)
        elbow_flex_l = get_val(row, headers, 'elbow_flex_l', 0)
        pro_sup_l = get_val(row, headers, 'pro_sup_l', 0)

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

        # === PELVIS (spine bone) - Using QUATERNION ===
        rig.pose.bones['spine'].location.x = -pelvis_tz
        rig.pose.bones['spine'].location.y = pelvis_ty
        rig.pose.bones['spine'].location.z = pelvis_tx
        rig.pose.bones['spine'].keyframe_insert(data_path="location", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['spine'],
                               -pelvis_tilt, pelvis_rotation, pelvis_list, 'YZX')
        rig.pose.bones['spine'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === RIGHT LEG ===
        set_bone_rotation_quat(rig.pose.bones['thigh.R'],
                               -hip_flexion_r, -hip_rotation_r, -hip_adduction_r, 'YZX')
        rig.pose.bones['thigh.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['shin.R'], knee_angle_r, 0, 0, 'YZX')
        rig.pose.bones['shin.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['foot.R'], -ankle_angle_r, 0, 0, 'YZX')
        rig.pose.bones['foot.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LEFT LEG ===
        set_bone_rotation_quat(rig.pose.bones['thigh.L'],
                               -hip_flexion_l, hip_rotation_l, hip_adduction_l, 'YZX')
        rig.pose.bones['thigh.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['shin.L'], knee_angle_l, 0, 0, 'YZX')
        rig.pose.bones['shin.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['foot.L'], -ankle_angle_l, 0, 0, 'YZX')
        rig.pose.bones['foot.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LUMBAR (spine.001) ===
        set_bone_rotation_quat(rig.pose.bones['spine.001'],
                               -lumbar_ext, lumbar_rot, lumbar_bend, 'YZX')
        rig.pose.bones['spine.001'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === THORAX (spine.002) ===
        set_bone_rotation_quat(rig.pose.bones['spine.002'],
                               -thorax_ext, thorax_rot, thorax_bend, 'YZX')
        rig.pose.bones['spine.002'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === RIGHT ARM ===
        set_bone_rotation_quat(rig.pose.bones['upper_arm.R'],
                               -arm_flex_r, -arm_rot_r, -arm_add_r, 'YZX')
        rig.pose.bones['upper_arm.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['forearm.R'],
                               -elbow_flex_r, -pro_sup_r, 0, 'YZX')
        rig.pose.bones['forearm.R'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        # === LEFT ARM ===
        set_bone_rotation_quat(rig.pose.bones['upper_arm.L'],
                               -arm_flex_l, arm_rot_l, arm_add_l, 'YZX')
        rig.pose.bones['upper_arm.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        set_bone_rotation_quat(rig.pose.bones['forearm.L'],
                               -elbow_flex_l, pro_sup_l, 0, 'YZX')
        rig.pose.bones['forearm.L'].keyframe_insert(data_path="rotation_quaternion", frame=frame)

        frame += 1
        if frame % 200 == 0:
            print(f"  Frame {frame}...")

    bpy.ops.object.mode_set(mode='OBJECT')

    # Set scene frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frame - 1
    bpy.context.scene.render.fps = target_fps

    print(f"  Applied {frame} frames using QUATERNION rotations")
    return frame


def cleanup_scene(rig_name):
    """Remove objects not related to the skeleton export."""
    keep_names = set()
    rig = bpy.data.objects.get(rig_name)
    if rig:
        keep_names.add(rig.name)
        for obj in bpy.data.objects:
            if obj.parent == rig and obj.type == 'MESH':
                keep_names.add(obj.name)

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

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in ['ARMATURE', 'MESH']:
            obj.select_set(True)

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    print(f"  Scene frame range: {frame_start} - {frame_end} ({frame_end - frame_start + 1} frames)")

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            print(f"  Action '{action.name}' range: {action.frame_range[0]:.0f} - {action.frame_range[1]:.0f}")
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
    print("Using QUATERNION rotations to avoid gimbal lock")

    if args.rig not in bpy.data.objects:
        print(f"ERROR: Armature '{args.rig}' not found!")
        print(f"Available armatures: {[o.name for o in bpy.data.objects if o.type == 'ARMATURE']}")
        return

    cleanup_scene(args.rig)
    apply_motion(args.rig, args.mot, target_fps=args.fps)
    export_fbx(args.output)

    print("Done!")


if __name__ == '__main__':
    main()
