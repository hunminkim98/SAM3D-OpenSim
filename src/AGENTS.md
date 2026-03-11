<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-10 10:54:42 KST -->

# src

## Purpose
This directory contains the core pipeline modules. It wraps SAM3D Body inference, stages reusable pipeline steps, post-processes MHR70 keypoints, transforms coordinates into the OpenSim frame, converts keypoints into marker trajectories, writes TRC files, estimates MoGe-based scene ground, builds runtime OpenSim marker/task specs, and provides reusable inverse kinematics, Blender export, and post-IK correction helpers.

## Key Files
| File | Description |
|------|-------------|
| `sam3d_inference.py` | High-level SAM3D Body wrapper that loads the model, manages single-person selection/tracking, and captures scene-ground metadata. |
| `inference_stage.py` | Shared Stage 1 orchestration for frame extraction, SAM3D inference, and canonical artifact writing. |
| `export_stage.py` | Shared Stage 2 orchestration for post-processing, coordinate transform, TRC export, IK, and FBX execution. |
| `pipeline_artifacts.py` | Canonical helpers for loading, extracting, and writing `video_outputs.json` and `inference_meta.json`. |
| `post_processing.py` | Temporal interpolation, Butterworth smoothing, bone normalization, and outlier utilities for keypoint sequences. |
| `coordinate_transform.py` | Camera-to-OpenSim axis transform, scaling, pelvis centering, contact-aware ground alignment, hybrid vertical translation, and `cam_t` global translation logic. |
| `keypoint_converter.py` | MHR70-to-OpenSim marker conversion with direct and derived marker support. |
| `moge_scene_ground.py` | MoGe point-cloud helpers for support-plane estimation and foot clearance measurement. |
| `opensim_marker_spec.py` | Centralized runtime OpenSim marker placements, weights, and IK task XML generation. |
| `trc_exporter.py` | TRC file writer and loader helpers for OpenSim-compatible marker trajectories. |
| `opensim_ik.py` | Reusable OpenSim/Pose2Sim inverse kinematics wrapper plus canonical external IK runner for the Pose2Sim environment. |
| `blender_export.py` | Shared Blender subprocess wrapper used by Stage 2 and the full pipeline. |
| `post_ik_foot_snap.py` | Post-IK MOT correction that lowers `pelvis_ty` on stance frames using exported contact metadata. |
| `__init__.py` | Lightweight package entry point that avoids importing heavy numerical dependencies eagerly. |

## Subdirectories
This directory has no nested versioned source directories.

## For AI Agents

### Working In This Directory
- Preserve array shape conventions such as `(T, K, 3)` for sequences and `(K, 3)` for single frames.
- Keep MHR70 index assumptions synchronized across converter, transformer, and post-processing code.
- Be careful with units: most internal geometry is normalized in meters and later converted to millimeters for TRC export.
- Coordinate and contact changes are high-risk because they affect TRC generation, IK quality, post-IK foot correction, and Blender exports simultaneously.
- Keep orchestration helpers separate from numerical kernels: `*_stage.py` and artifact modules should compose domain modules, not absorb their math.

### Testing Requirements
- Run a syntax check such as `python -m py_compile src/*.py utils/*.py` after editing core modules.
- Smoke-test `run_export.py` against a saved `video_outputs.json` after changing `post_processing.py`, `coordinate_transform.py`, `keypoint_converter.py`, or `trc_exporter.py`.
- Re-run `test_imports.py` and an inference smoke test after changing `sam3d_inference.py`.
- Re-run an IK export on Windows after modifying `opensim_ik.py`, `opensim_marker_spec.py`, `post_ik_foot_snap.py`, or marker semantics.

### Common Patterns
- Modules are intentionally NumPy-centric and avoid heavy class graphs outside the main wrappers.
- Derived markers are computed from anatomical midpoints or interpolated landmarks.
- Coordinate transformation applies axis remapping first, then subject scaling, then pelvis centering or global translation, then ground alignment and optional hybrid vertical correction.
- Inference now emits both per-frame outputs and clip-level metadata such as selected-person state, support-surface mode, and scene-ground plane estimates.
- Runtime OpenSim marker/task generation is centralized so direct export, full-pipeline IK, and fallback IK stay consistent.
- Canonical stage boundaries are explicit: Stage 1 produces artifact JSON, Stage 2 consumes artifact JSON. Cross-stage data exchange should go through `pipeline_artifacts.py` rather than ad hoc dicts in CLI files.

## Dependencies

### Internal
- `utils/` for config loading and video metadata.
- `config/marker_mapping.yaml` and `models/Markers_MHR70.xml` for marker semantics, weights, and foot marker offsets.
- `run_inference.py`, `run_export.py`, and `run_pipeline.py` as the primary callers.

### External
- NumPy throughout the directory.
- SciPy in `post_processing.py` and `coordinate_transform.py` for filtering and smoothing.
- Torch and upstream SAM3D Body packages in `sam3d_inference.py`.
- MoGe in `moge_scene_ground.py` and `sam3d_inference.py`.
- OpenSim/Pose2Sim in `opensim_ik.py`, `opensim_marker_spec.py`, and `post_ik_foot_snap.py`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
