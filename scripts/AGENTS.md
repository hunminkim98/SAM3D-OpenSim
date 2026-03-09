<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# scripts

## Purpose
This directory contains Blender-side export scripts used to turn TRC or OpenSim MOT output into animated FBX or GLB assets. These scripts are tightly coupled to Blender's Python API and to the rig assumptions encoded in the repository's `.blend` template.

## Key Files
| File | Description |
|------|-------------|
| `export_fbx_blender.py` | Builds and animates a simple armature directly from TRC marker trajectories, then exports FBX. |
| `export_fbx_skely.py` | Applies OpenSim `.mot` joint angles to the `metarig_skely` rig in the root Blender file and exports FBX. |
| `export_fbx_skely_quat.py` | Quaternion-based variant of the Skely exporter that also writes GLB to avoid Euler wrapping artifacts. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- These scripts are meant to run inside Blender, so keep `bpy` and `mathutils` imports isolated to Blender execution contexts.
- Preserve the argument-parsing convention that reads parameters after `--`; repository runners rely on it.
- Coordinate and rig changes must stay aligned with `Import_OS4_Patreon_Aitor_Skely.blend`, OpenSim MOT column names, and the viewer expectations in `index.html`.

### Testing Requirements
- Validate script syntax first with ordinary Python if the change is import-safe.
- Perform functional verification in Blender with representative `.trc` or `.mot` input after changing rotations, rig names, or export settings.
- For quaternion export changes, inspect both generated `.fbx` and `.glb` playback in Blender or the local viewer.

### Common Patterns
- TRC-driven export reconstructs a lightweight armature from marker positions.
- MOT-driven export maps OpenSim DOFs onto named bones in the Skely rig.
- Angle preprocessing focuses on unwrapping discontinuities rather than changing the source kinematics.

## Dependencies

### Internal
- Root rig template: `Import_OS4_Patreon_Aitor_Skely.blend`.
- Outputs produced by `run_export.py` and `src/trc_exporter.py`.
- Viewer files `index.html` and `server.py` for inspecting exported models.

### External
- Blender 5.x with `bpy` and `mathutils`.
- OpenSim-generated `.mot` files and OpenSim/TRC coordinate conventions.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
