"""
Blender FBX Export Script
=========================

This script is run inside Blender to export skeleton animation to FBX.
It reads TRC marker data and creates an animated armature.

Usage:
    blender --background --python export_fbx_blender.py -- --trc markers.trc --output skeleton.fbx
"""

import sys
import argparse
import math

# Parse arguments after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--trc", required=True, help="Input TRC file")
parser.add_argument("--output", required=True, help="Output FBX file")
parser.add_argument("--fps", type=float, default=30.0, help="Animation FPS")
parser.add_argument("--scale", type=float, default=0.001, help="Scale factor (mm to m)")
args = parser.parse_args(argv)

import bpy
import mathutils


def load_trc(trc_path):
    """Load TRC file and return marker data."""
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    header_values = lines[2].strip().split('\t')
    fps = float(header_values[0])
    num_frames = int(header_values[2])
    num_markers = int(header_values[3])

    # Parse marker names
    marker_line = lines[3].strip().split('\t')
    marker_names = []
    for i in range(2, len(marker_line), 3):
        if marker_line[i].strip():
            marker_names.append(marker_line[i].strip())

    # Parse data
    markers = {}
    for name in marker_names:
        markers[name] = []

    for line in lines[6:]:
        if not line.strip():
            continue
        values = line.strip().split('\t')
        for j, name in enumerate(marker_names):
            idx = 2 + j * 3
            x = float(values[idx])
            y = float(values[idx + 1])
            z = float(values[idx + 2])
            markers[name].append((x, y, z))

    return markers, marker_names, fps, num_frames


def create_armature(marker_names):
    """Create armature with bones for key body parts."""
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "SAM3D_Skeleton"
    arm_data = armature.data
    arm_data.name = "SAM3D_Armature"

    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Define skeleton hierarchy
    bones_def = [
        # (name, parent, head_marker, tail_marker or offset)
        ("Hips", None, "PelvisCenter", (0, 0.1, 0)),
        ("Spine", "Hips", "SpineMid", (0, 0.15, 0)),
        ("Chest", "Spine", "Thorax", (0, 0.1, 0)),
        ("Neck", "Chest", "Neck", (0, 0.08, 0)),
        ("Head", "Neck", "Nose", (0, 0.1, 0)),

        # Left leg
        ("LeftUpLeg", "Hips", "LHip", "LKnee"),
        ("LeftLeg", "LeftUpLeg", "LKnee", "LAnkle"),
        ("LeftFoot", "LeftLeg", "LAnkle", "LHeel"),
        ("LeftToes", "LeftFoot", "LBigToe", (0.05, 0, 0)),

        # Right leg
        ("RightUpLeg", "Hips", "RHip", "RKnee"),
        ("RightLeg", "RightUpLeg", "RKnee", "RAnkle"),
        ("RightFoot", "RightLeg", "RAnkle", "RHeel"),
        ("RightToes", "RightFoot", "RBigToe", (0.05, 0, 0)),

        # Left arm
        ("LeftShoulder", "Chest", "LAcromion", "LShoulder"),
        ("LeftArm", "LeftShoulder", "LShoulder", "LElbow"),
        ("LeftForeArm", "LeftArm", "LElbow", "LWrist"),
        ("LeftHand", "LeftForeArm", "LWrist", (0.08, 0, 0)),

        # Right arm
        ("RightShoulder", "Chest", "RAcromion", "RShoulder"),
        ("RightArm", "RightShoulder", "RShoulder", "RElbow"),
        ("RightForeArm", "RightArm", "RElbow", "RWrist"),
        ("RightHand", "RightForeArm", "RWrist", (0.08, 0, 0)),
    ]

    bones = {}
    for bone_name, parent_name, head_marker, tail_info in bones_def:
        bone = arm_data.edit_bones.new(bone_name)
        bone.head = (0, 0, 0)

        if isinstance(tail_info, tuple):
            bone.tail = tail_info
        else:
            bone.tail = (0, 0.1, 0)  # Default, will be updated

        if parent_name and parent_name in bones:
            bone.parent = bones[parent_name]
            bone.use_connect = False

        bones[bone_name] = bone

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature, bones_def


def animate_armature(armature, markers, bones_def, fps, scale):
    """Animate armature bones based on marker positions."""
    bpy.context.view_layer.objects.active = armature

    # Set animation settings
    bpy.context.scene.render.fps = int(fps)

    num_frames = len(markers.get("PelvisCenter", markers.get("LHip", [[0,0,0]])))
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

    # Create action
    if not armature.animation_data:
        armature.animation_data_create()

    action = bpy.data.actions.new(name="SAM3D_Motion")
    armature.animation_data.action = action

    bpy.ops.object.mode_set(mode='POSE')

    for frame in range(num_frames):
        bpy.context.scene.frame_set(frame + 1)

        for bone_name, parent_name, head_marker, tail_info in bones_def:
            if bone_name not in armature.pose.bones:
                continue

            pose_bone = armature.pose.bones[bone_name]

            # Get head position from marker
            if head_marker in markers and frame < len(markers[head_marker]):
                pos = markers[head_marker][frame]
                # Convert from mm to m and apply coordinate transform
                # TRC: X=forward, Y=up, Z=right -> Blender: X=right, Y=forward, Z=up
                x = pos[2] * scale  # Z -> X
                y = pos[0] * scale  # X -> Y
                z = pos[1] * scale  # Y -> Z

                # Set bone location (for root bone)
                if bone_name == "Hips":
                    pose_bone.location = (x, y, z)
                    pose_bone.keyframe_insert(data_path="location", frame=frame + 1)

        # Calculate rotations based on marker positions
        for bone_name, parent_name, head_marker, tail_info in bones_def:
            if bone_name not in armature.pose.bones:
                continue

            pose_bone = armature.pose.bones[bone_name]

            if isinstance(tail_info, str) and tail_info in markers:
                if head_marker in markers and frame < len(markers[head_marker]):
                    head_pos = markers[head_marker][frame]
                    tail_pos = markers[tail_info][frame]

                    # Calculate direction vector
                    hx, hy, hz = head_pos[2]*scale, head_pos[0]*scale, head_pos[1]*scale
                    tx, ty, tz = tail_pos[2]*scale, tail_pos[0]*scale, tail_pos[1]*scale

                    direction = mathutils.Vector((tx - hx, ty - hy, tz - hz))
                    if direction.length > 0.001:
                        direction.normalize()

                        # Calculate rotation to align bone with direction
                        up = mathutils.Vector((0, 0, 1))
                        rotation = direction.rotation_difference(up)

                        pose_bone.rotation_mode = 'QUATERNION'
                        pose_bone.rotation_quaternion = rotation
                        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame + 1)

    bpy.ops.object.mode_set(mode='OBJECT')


def export_fbx(armature, output_path):
    """Export armature to FBX."""
    # Select only the armature
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Export
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE'},
        use_armature_deform_only=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
    )
    print(f"Exported FBX to: {output_path}")


def main():
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Load TRC
    print(f"Loading TRC: {args.trc}")
    markers, marker_names, fps, num_frames = load_trc(args.trc)
    print(f"  Markers: {len(marker_names)}, Frames: {num_frames}, FPS: {fps}")

    # Use TRC fps if not overridden
    if args.fps == 30.0:
        args.fps = fps

    # Create armature
    print("Creating armature...")
    armature, bones_def = create_armature(marker_names)

    # Animate
    print("Animating...")
    animate_armature(armature, markers, bones_def, args.fps, args.scale)

    # Export
    print(f"Exporting to: {args.output}")
    export_fbx(armature, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
