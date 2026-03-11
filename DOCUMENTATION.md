# SAM3D-OpenSim Technical Documentation

Compact technical reference for the current SAM3D-OpenSim pipeline.

For the authoritative long-form reference, see `docs/FULL_DOCUMENTATION.md`.

## Architecture Overview

The repository now centers on a canonical two-stage flow:

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

Wrapper entrypoints:

- `run_inference.py`: canonical Stage 1 CLI
- `run_export.py`: canonical Stage 2 CLI
- `run_pipeline.py`: thin in-process wrapper over the shared Stage 1 + Stage 2 composition helper
- `run_full_pipeline.py`: thin Windows-oriented subprocess wrapper that runs Stage 1 first, then Stage 2

Core modules:

- `src/inference_stage.py`: shared Stage 1 orchestration
- `src/export_stage.py`: shared Stage 2 orchestration
- `src/pipeline_artifacts.py`: artifact IO helpers for `video_outputs.json` and `inference_meta.json`
- `src/pipeline_runner.py`: shared in-process Stage 1 + Stage 2 composition
- `src/subprocess_pipeline_runner.py`: shared subprocess Stage 1 + Stage 2 composition
- `src/pipeline_runtime_common.py`: shared runtime option resolution, output-directory handling, and summary formatting
- `src/sam3d_inference.py`: SAM3D Body wrapper, single-person tracking, support-surface flow
- `src/coordinate_transform.py`: coordinate transform, ground alignment, vertical translation
- `src/moge_scene_ground.py`: MoGe scene-ground and support-plane helpers
- `src/opensim_ik.py`: OpenSim and Pose2Sim execution helpers
- `src/post_ik_foot_snap.py`: stance-phase MOT correction after IK
- `src/blender_export.py`: shared Blender export subprocess helper

## Coordinate Systems

### SAM3D Body (camera space)

- X: right
- Y: down
- Z: forward

### OpenSim (world space)

- X: forward
- Y: up
- Z: right

### Transform

```python
CAMERA_TO_OPENSIM = np.array([
    [0,  0,  1],
    [0, -1,  0],
    [1,  0,  0],
])
```

## Marker Pipeline

Do not conflate these three layers:

| Layer | Count | Meaning |
|------|------:|---------|
| Raw MHR70 keypoints | 70 | Direct SAM3D Body outputs |
| Exported TRC markers | 73 | 70 direct markers plus 3 derived markers |
| OpenSim IK runtime markers | 28 | Runtime subset used by the current IK task set |

Derived TRC markers:

- `PelvisCenter`
- `Thorax`
- `SpineMid`

The current runtime IK subset includes:

- Head: `Nose`, `LEye`, `REye`, `LEar`, `REar`
- Torso: `Neck`, `LShoulder`, `RShoulder`
- Arms and hands: `LElbow`, `RElbow`, `LWrist`, `RWrist`, `LIndex3`, `RIndex3`, `LMiddleTip`, `RMiddleTip`
- Legs: `LHip`, `RHip`, `LKnee`, `RKnee`, `LAnkle`, `RAnkle`
- Feet: `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel`

The default source of truth for direct marker naming, derived markers, runtime IK subset order, and runtime IK weight overrides is `config/marker_mapping.yaml`.

## Global Translation

`pred_cam_t` is the camera translation signal emitted by SAM3D Body.

When `--global-translation` is enabled:

1. `pred_cam_t` is transformed from camera space to OpenSim space
2. the signal is smoothed
3. X/Z motion is always used
4. Y motion may remain legacy X/Z-only or switch to hybrid support-plane blending

Best results still depend on `--fov moge2`.

## Ground Alignment and Vertical Translation

### Ground alignment modes

| Mode | Behavior |
|------|----------|
| `per_frame_snap` | Legacy mode, lowest foot point is snapped to ground every frame |
| `contact_aware` | Preserve airborne motion, re-anchor during stance only |
| `auto` | Choose between the above based on the motion signal |

### Vertical translation modes

| Mode | Behavior |
|------|----------|
| `legacy_xz_only` | Only horizontal translation from `pred_cam_t` |
| `hybrid_support_plane` | Blend support-plane clearance with `pred_cam_t_y` |
| `auto` | Use hybrid mode when support-plane confidence is good |

### Support surface

Support-surface grounding uses MoGe scene geometry.

- `auto`: estimate the support plane automatically
- `manual_roi`: let the user draw and accept a support-surface ROI before inference continues

## Post-IK Foot Snap

Post-IK foot snap is a MOT-level correction.

It:

- preserves the raw IK result as `*_ik_raw.mot`
- lowers `pelvis_ty` during stance frames only
- writes a `*_foot_snap_report.json`
- does not rerun marker fitting

## File Formats

### `video_outputs.json`

Per-frame SAM3D export:

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

Optional fields may also include `pred_keypoints_2d`, `shape_params`, and per-person `scene_ground`.

### TRC export

The canonical export path currently writes 73 markers in millimeters.

### MOT export

OpenSim IK writes joint coordinates to `*_ik.mot`. If post-IK foot snap is enabled, the raw file is preserved separately as `*_ik_raw.mot`.

### Graph export

When `--save_graph` is enabled:

- `graphs/coords/` stores one PNG per TRC marker, with X/Y/Z plotted against time
- `graphs/angles/` stores one PNG per MOT coordinate column, when IK is enabled

## Output Files

Typical output directory:

```text
output_YYYYMMDD_HHMMSS_<video>/
├── frames/                        # Extracted JPG frames
├── video_outputs.json             # Canonical Stage 1 outputs
├── inference_meta.json            # Stage 1 metadata
├── post_ik_contact_meta.json      # Contact/flight metadata saved before IK
├── markers_*.trc                  # OpenSim marker trajectories
├── *_ik.mot                       # IK result
├── *_ik_raw.mot                   # Raw IK result when post-IK correction is enabled
├── *_foot_snap_report.json        # Post-IK correction report when enabled
├── *_model.osim                   # Model with runtime markers
├── graphs/coords/*.png            # TRC coordinate plots when --save_graph is enabled
├── graphs/angles/*.png            # MOT angle plots when --save_graph is enabled and IK runs
└── *.fbx                          # Blender animation export
```

Additional convenience-wrapper artifacts:

- `run_pipeline.py` may also emit `keypoints_raw.json`
- `run_pipeline.py` may also emit `processing_report.json`

## Validation Checklist

After orchestration or CLI changes:

```bash
python run_inference.py --help
python run_export.py --help
python run_full_pipeline.py --help
```

After export-stage changes:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 --skip-ik --skip-fbx
```

`run_pipeline.py --visualize` is currently a deprecated compatibility flag and does not produce extra viewer or media outputs.
