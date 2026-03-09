<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# src

## Purpose
This directory contains the core pipeline modules. It wraps SAM3D Body inference, post-processes MHR70 keypoints, transforms coordinates into the OpenSim frame, converts keypoints into marker trajectories, writes TRC files, and provides reusable inverse kinematics helpers.

## Key Files
| File | Description |
|------|-------------|
| `sam3d_inference.py` | High-level SAM3D Body wrapper that loads the model and optional detector, segmentor, and FOV components. |
| `post_processing.py` | Temporal interpolation, Butterworth smoothing, bone normalization, and outlier utilities for keypoint sequences. |
| `coordinate_transform.py` | Camera-to-OpenSim axis transform, scaling, pelvis centering, ground alignment, and `cam_t` global translation logic. |
| `keypoint_converter.py` | MHR70-to-OpenSim marker conversion with direct and derived marker support. |
| `trc_exporter.py` | TRC file writer and loader helpers for OpenSim-compatible marker trajectories. |
| `opensim_ik.py` | Reusable OpenSim/Pose2Sim inverse kinematics wrapper and fallback logic. |
| `__init__.py` | Package marker for source-module imports. |

## Subdirectories
This directory has no nested versioned source directories.

## For AI Agents

### Working In This Directory
- Preserve array shape conventions such as `(T, K, 3)` for sequences and `(K, 3)` for single frames.
- Keep MHR70 index assumptions synchronized across converter, transformer, and post-processing code.
- Be careful with units: most internal geometry is normalized in meters and later converted to millimeters for TRC export.
- Coordinate changes are high-risk because they affect TRC generation, IK quality, and Blender exports simultaneously.

### Testing Requirements
- Run a syntax check such as `python -m py_compile src/*.py utils/*.py` after editing core modules.
- Smoke-test `run_export.py` against a saved `video_outputs.json` after changing `post_processing.py`, `coordinate_transform.py`, `keypoint_converter.py`, or `trc_exporter.py`.
- Re-run `test_imports.py` and an inference smoke test after changing `sam3d_inference.py`.
- Re-run an IK export on Windows after modifying `opensim_ik.py` or marker semantics.

### Common Patterns
- Modules are intentionally NumPy-centric and avoid heavy class graphs outside the main wrappers.
- Derived markers are computed from anatomical midpoints or interpolated landmarks.
- Coordinate transformation applies axis remapping first, then subject scaling, then pelvis centering or global translation, then ground alignment.

## Dependencies

### Internal
- `utils/` for config loading and video metadata.
- `config/marker_mapping.yaml` and `models/Markers_MHR70.xml` for marker semantics.
- `run_inference.py`, `run_export.py`, and `run_pipeline.py` as the primary callers.

### External
- NumPy throughout the directory.
- SciPy in `post_processing.py` and `coordinate_transform.py` for filtering and smoothing.
- Torch and upstream SAM3D Body packages in `sam3d_inference.py`.
- OpenSim/Pose2Sim in `opensim_ik.py`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
