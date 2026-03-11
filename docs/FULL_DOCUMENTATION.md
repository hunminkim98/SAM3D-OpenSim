# SAM3D-OpenSim - Full Documentation

**Version:** 2.2.0  
**Last Updated:** 2026-03-10  
**Purpose:** Technical and user-facing reference for the current SAM3D-OpenSim pipeline.

---

## 1. Overview

SAM3D-OpenSim converts monocular video into OpenSim-ready motion data using SAM3D Body for 3D pose estimation. The repository now centers on a canonical two-stage workflow:

1. `run_inference.py`
2. `run_export.py`

This split is intentional:

- Stage 1 is expensive and GPU-bound
- Stage 2 is fast and should be re-run repeatedly while tuning export settings

`run_full_pipeline.py` is a thin subprocess orchestrator that runs those two stages in sequence. `run_pipeline.py` is a thin in-process convenience wrapper that composes the same shared stage helpers in one process.

### Current Capabilities

- Direct 3D pose from SAM3D Body
- 70-keypoint MHR70 skeleton
- Optional `sam3`, `vitdet`, `yolo11`, or `none` detector
- Optional MoGe2 FOV estimation
- Single-person interactive selection and tracking
- Optional manual support-surface ROI selection
- Contact-aware ground alignment
- Hybrid support-plane vertical translation
- OpenSim IK with centralized runtime marker/task generation
- Optional post-IK stance-phase foot snap
- Blender FBX export

---

## 2. Pipeline Architecture

### Canonical Flow

```text
Video
  -> run_inference.py
     -> frames/
     -> video_outputs.json
     -> inference_meta.json
  -> run_export.py
     -> post_ik_contact_meta.json
     -> markers_*.trc
     -> *_ik.mot
     -> *.fbx
```

### Entry Points

| Entry Point | Role |
|-------------|------|
| `run_inference.py` | Canonical Stage 1 implementation |
| `run_export.py` | Canonical Stage 2 implementation |
| `run_pipeline.py` | Thin one-process convenience wrapper over Stage 1 + Stage 2 |
| `run_full_pipeline.py` | Thin Windows-oriented subprocess orchestrator that runs Stage 1 in `sam_3d_body` then Stage 2 in the current environment |

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `src/inference_stage.py` | Shared Stage 1 orchestration |
| `src/export_stage.py` | Shared Stage 2 orchestration |
| `src/pipeline_artifacts.py` | Canonical artifact loading/writing helpers |
| `src/pipeline_runner.py` | Shared in-process Stage 1 + Stage 2 composition |
| `src/subprocess_pipeline_runner.py` | Shared subprocess Stage 1 + Stage 2 composition |
| `src/pipeline_runtime_common.py` | Shared runtime option resolution, output-directory handling, and summary formatting |
| `src/sam3d_inference.py` | SAM3D Body wrapper, tracking, selection UI, scene-ground capture |
| `src/coordinate_transform.py` | Coordinate transform, ground alignment, vertical translation |
| `src/moge_scene_ground.py` | Support-plane estimation and foot-clearance helpers |
| `src/opensim_ik.py` | Direct and external OpenSim IK execution |
| `src/blender_export.py` | Shared Blender export subprocess helper |
| `src/post_ik_foot_snap.py` | MOT-level stance correction after IK |

---

## 3. Environments and External Tools

### Conda Environments

| Environment | Purpose |
|-------------|---------|
| `sam_3d_body` | SAM3D Body inference, SAM3, MoGe2 |
| `Pose2Sim` | OpenSim IK and post-IK MOT correction |

### External Components

| Component | Expected Source |
|-----------|-----------------|
| SAM3D Body | `C:\Sam3dBody\sam-3d-body` |
| SAM3 | `C:\Sam3dBody\sam3` |
| MoGe2 | Installed in `sam_3d_body` |
| Pose2Sim | Installed in `Pose2Sim` |
| Blender | Auto-detected under `C:\Program Files\Blender Foundation\Blender *\blender.exe` or overridden by env var |

### Auto-Discovery

The repository includes Windows path discovery helpers. If your setup is non-standard, the following overrides are supported:

- `SAM3D_PYTHON` or `SAM3D_OPENSIM_SAM3D_PYTHON`
- `OPENSIM_PYTHON` or `SAM3D_OPENSIM_OPENSIM_PYTHON`
- `POSE2SIM_SETUP` or `SAM3D_OPENSIM_POSE2SIM_SETUP`
- `BLENDER_PATH` or `SAM3D_OPENSIM_BLENDER_PATH`

---

## 4. Installation and Verification

### Verify Inference Dependencies

```bash
conda activate sam_3d_body
python test_imports.py
```

For `sam3`, the common missing dependencies are:

```bash
pip install decord psutil ftfy regex triton-windows
```

### Verify CLI Surface

```bash
python run_inference.py --help
python run_export.py --help
python run_full_pipeline.py --help
```

### Verify Export Logic Without Re-Running Inference

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 --skip-ik --skip-fbx
```

---

## 5. Usage

### 5.1 Best-Quality Full Pipeline

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### 5.2 Full Pipeline With Manual Support Surface

```bash
python run_full_pipeline.py --input jump.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation \
    --support-surface-mode manual_roi \
    --vertical-translation-mode hybrid_support_plane \
    --post-ik-foot-snap stance_only
```

### 5.3 Two-Stage Workflow

Stage 1:

```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

Stage 2:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation \
    --ground-alignment-mode contact_aware \
    --vertical-translation-mode hybrid_support_plane \
    --save_graph
```

### 5.4 Re-Export Existing SAM3D Outputs

```bash
python run_export.py --input C:\Sam3dBody\sam-3d-body\outputs\my_video\video_outputs.json \
    --height 1.75
```

### 5.5 CPU Inference

`run_full_pipeline.py` does not expose `--device`. For CPU mode, use Stage 1 directly:

```bash
python run_inference.py --input video.mp4 --device cpu
python run_export.py --input output_dir/video_outputs.json --height 1.69
```

---

## 6. Important CLI Flags

### Stage 1 / Full-Pipeline Inference Flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--detector` | `vitdet`, `yolo11`, `sam3`, `none` | `vitdet` |
| `--segmentor` | `sam2`, `none` | `none` |
| `--fov` | `moge2`, `none` | `moge2` |
| `--single_person` | Interactive single-person tracking | `true` |
| `--support-surface-mode` | `auto`, `manual_roi` | config default |
| `--use-mask` | Enable segmentation-conditioned inference | `false` |

### Stage 2 / Export Flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--global-translation` | Use `pred_cam_t` for translation | `false` |
| `--smooth` | Butterworth cutoff in Hz | `6.0` |
| `--ground-alignment-mode` | `auto`, `contact_aware`, `per_frame_snap` | `auto` |
| `--vertical-translation-mode` | `auto`, `legacy_xz_only`, `hybrid_support_plane` | `auto` |
| `--post-ik-foot-snap` | `off`, `auto`, `stance_only` | `off` |
| `--save_graph` | Save TRC coordinate and MOT angle plots under `graphs/` | `false` |
| `--skip-ik` | Stop after TRC export | `false` |
| `--skip-fbx` | Stop after IK | `false` |

---

## 7. Stage 1 Artifacts

### `video_outputs.json`

Canonical per-frame SAM3D export:

```json
[
  {
    "frame": "frame_000000.jpg",
    "outputs": [
      {
        "bbox": [x1, y1, x2, y2],
        "focal_length": 1234.56,
        "pred_keypoints_3d": [[x, y, z], "..."],
        "pred_cam_t": [tx, ty, tz]
      }
    ]
  }
]
```

Optional fields may also include:

- `pred_keypoints_2d`
- `shape_params`
- per-person `scene_ground`

### `inference_meta.json`

Typical contents:

- input video path
- FPS used for extraction
- video metadata
- `single_person`
- `support_surface_mode`
- selection metadata
- clip-level scene-ground metadata
- optional `vertical_translation_mode`

---

## 8. Support Surface and Scene Ground

### Auto Mode

In auto mode, the pipeline estimates a support plane from MoGe scene points, usually by fitting a floor-like plane from scene geometry.

### Manual ROI Mode

In `manual_roi` mode:

1. You choose the tracked person
2. You draw a support-surface ROI
3. The pipeline previews the fitted plane
4. Inference does not continue until the preview is explicitly accepted

This is intended for scenes where the true contact surface is not the room floor, for example:

- treadmill decks
- boxes or platforms
- elevated takeoff areas

---

## 9. Ground Alignment and Vertical Translation

### Ground Alignment Modes

| Mode | Behavior |
|------|----------|
| `per_frame_snap` | Legacy mode, lowest foot is snapped to ground every frame |
| `contact_aware` | Preserve airborne motion and re-anchor during stance |
| `auto` | Choose between the above based on available motion cues |

### Vertical Translation Modes

| Mode | Behavior |
|------|----------|
| `legacy_xz_only` | Do not use support-plane hybrid Y correction |
| `hybrid_support_plane` | Blend support-plane cues and `pred_cam_t_y` |
| `auto` | Use hybrid mode when scene-ground data is available, otherwise legacy mode |

### Practical Interpretation

- `contact_aware` is the preferred mode for jumps
- `hybrid_support_plane` is the preferred vertical mode when MoGe support-plane data is trusted
- `per_frame_snap` is mainly a compatibility fallback

---

## 10. Marker Conversion and IK

The marker pipeline has three layers that should not be conflated:

| Layer | Count | Meaning |
|------|------:|---------|
| Raw MHR70 keypoints | 70 | Direct SAM3D Body outputs before marker conversion |
| Exported TRC markers | 73 | 70 direct marker names plus 3 derived markers |
| OpenSim IK runtime markers | 28 | The subset actually used by the IK task set |

### Raw MHR70 Keypoints

SAM3D Body predicts the canonical MHR70 skeleton:

- 21 body and foot landmarks
- 21 right-hand landmarks
- 21 left-hand landmarks
- 7 extra anatomical landmarks

The canonical ordering is defined in `src/keypoint_converter.py`, and the converter maps those 70 raw keypoints to 70 direct export marker names.

### TRC Marker Export

The canonical export path writes 73 TRC markers:

- 70 direct markers from the raw MHR70 keypoints
- `PelvisCenter`
- `Thorax`
- `SpineMid`

These three derived markers are created during export and are not additional SAM3D keypoints.

### OpenSim IK Marker Subset

OpenSim IK does not use all 73 exported TRC markers. The current runtime task set uses the following 28-marker subset:

| Group | Runtime IK Markers |
|-------|--------------------|
| Head | `Nose`, `LEye`, `REye`, `LEar`, `REar` |
| Torso | `Neck`, `LShoulder`, `RShoulder` |
| Arms and hands | `LElbow`, `RElbow`, `LWrist`, `RWrist`, `LIndex3`, `RIndex3`, `LMiddleTip`, `RMiddleTip` |
| Legs | `LHip`, `RHip`, `LKnee`, `RKnee`, `LAnkle`, `RAnkle` |
| Feet | `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel` |

The default source of truth for runtime IK subset order and runtime IK weight overrides is `config/marker_mapping.yaml`.

### Why Foot Markers Matter

The added heel and toe markers help OpenSim IK fit lower-limb and foot posture better, especially in landing and stance phases.

---

## 11. Post-IK Foot Snap

Post-IK foot snap is a MOT-level correction, not a TRC correction.

It:

- preserves the raw IK result
- adjusts `pelvis_ty` during stance frames only
- leaves joint angles unchanged
- is meant to reduce visible foot hover without re-running IK

When enabled, the output directory may contain:

- `*_ik_raw.mot`
- corrected `*_ik.mot`
- `*_foot_snap_report.json`

---

## 12. FBX Export

FBX export uses:

- `Import_OS4_Patreon_Aitor_Skely.blend`
- `scripts/export_fbx_skely.py`
- a detected Blender executable

The exporter consumes the final MOT file, including any post-IK corrected `pelvis_ty`.

---

## 13. Output Files

| File | Meaning |
|------|---------|
| `frames/` | Extracted video frames |
| `video_outputs.json` | Canonical Stage 1 outputs |
| `inference_meta.json` | Stage 1 metadata |
| `post_ik_contact_meta.json` | Contact/flight metadata saved before IK |
| `markers_*.trc` | OpenSim marker trajectories |
| `*_ik.mot` | IK result |
| `*_ik_raw.mot` | Raw IK result when post-IK correction is enabled |
| `*_model.osim` | Model with runtime markers |
| `*_foot_snap_report.json` | Post-IK correction report |
| `graphs/coords/*.png` | TRC marker coordinate plots when `--save_graph` is enabled |
| `graphs/angles/*.png` | MOT coordinate plots when `--save_graph` is enabled and IK runs |
| `*.fbx` | Blender/export animation |
| `processing_report.json` | Combined pipeline summary from `run_pipeline.py` |

`run_pipeline.py` may also emit `keypoints_raw.json` for debugging or compatibility workflows.

---

## 14. Troubleshooting

### `No module named 'sam3'`

Use the `sam_3d_body` environment and ensure the SAM3 repo root is installed or auto-detected correctly. Common fix:

```bash
pip install -e C:\Sam3dBody\sam3
pip install decord psutil ftfy regex triton-windows
```

### `Could not load detector: No module named 'triton'`

On Windows:

```bash
pip install triton-windows
```

### MoGe2 Not Loading

Verify:

```bash
python -c "from moge.model.v2 import MoGeModel; print('ok')"
```

### Full Pipeline Shows Wrong Stage Boundaries

Use the current `run_full_pipeline.py`. It should call `run_inference.py` first and `run_export.py` second. If you still see Stage 1 producing TRC directly, you are likely running an outdated copy of the repository.

### Hover After IK

If stance frames still hover:

1. Use `--support-surface-mode manual_roi` for non-floor contact scenes
2. Use `--ground-alignment-mode contact_aware`
3. Use `--vertical-translation-mode hybrid_support_plane`
4. Enable `--post-ik-foot-snap stance_only`

### `graphs/angles/` Was Not Created

If `--skip-ik` was used, only `graphs/coords/` is expected.

Angle graphs require a generated MOT file.

### CUDA Out of Memory

Options:

- lower `--fps`
- use `run_inference.py --device cpu`

### OpenSim IK Failures

Check:

- the TRC file exists and opens
- the `Pose2Sim` environment is installed
- `post_ik_contact_meta.json` exists if post-IK correction is enabled

---

## 15. Testing Recommendations

### After Inference Integration Changes

```bash
conda activate sam_3d_body
python test_imports.py
python run_inference.py --help
```

### After Export or Coordinate Changes

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 --skip-ik --skip-fbx
```

`run_pipeline.py --visualize` is currently a deprecated compatibility flag and does not produce extra viewer or media outputs.

### After IK or Blender Changes

Validate on Windows with:

- working `Pose2Sim` environment
- detected Blender installation

The Linux sandbox is not a full substitute for those runtime checks.

---

## 16. Changelog Summary

### 2026-03-10

- Canonical two-stage architecture formalized around `run_inference.py` and `run_export.py`
- `run_full_pipeline.py` reduced to a thin orchestrator
- Shared stage helpers added in `src/inference_stage.py` and `src/export_stage.py`
- Shared artifact helpers added in `src/pipeline_artifacts.py`
- Shared in-process and subprocess runners added under `src/`
- Shared runtime option and summary helper added under `src/pipeline_runtime_common.py`
- Shared Blender export helper added
- External OpenSim IK path centralized
- Support-surface, hybrid vertical translation, and post-IK foot snap documented as first-class features
