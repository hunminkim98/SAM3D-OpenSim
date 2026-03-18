# SAM3D-OpenSim

Convert monocular video into OpenSim-ready motion data using SAM3D Body for 3D pose estimation.

## Overview

SAM3D-OpenSim takes a single video, runs SAM3D Body inference, converts the resulting MHR70 skeleton into OpenSim marker trajectories, solves inverse kinematics, and can export a rigged FBX animation.

The repository now follows a canonical two-stage pipeline:

```text
Video
  -> run_inference.py
     -> frames/
     -> video_outputs.json
     -> inference_meta.json
     -> mesh_vis/overlay.mp4         # optional Stage 1 mesh overlay video
     -> mesh_export/frame_*.ply      # optional Stage 1 mesh sequence
  -> run_export.py
     -> markers_*.trc
     -> *_ik.mot
     -> *.fbx
```

`run_full_pipeline.py` is a thin subprocess orchestrator over those two stages. `run_pipeline.py` is a thin in-process convenience wrapper that composes the same shared stage helpers inside one process.

The root entrypoints now delegate to shared orchestration helpers:

- `src/pipeline_runner.py` for in-process Stage 1 + Stage 2 composition
- `src/subprocess_pipeline_runner.py` for multi-environment subprocess orchestration
- `src/pipeline_runtime_common.py` for shared runtime option resolution, output-directory handling, and summary formatting

## Config.toml CLI

Install the repo in editable mode and run the pipeline through the new console command:

```bash
pip install -e .
sam3d-opensim --config Config.toml
```

`Config.toml` is the user-facing place to manage input paths, subject values, runtime mode, detector/FOV settings, smoothing, skip flags, and export defaults. The legacy root scripts still work and also accept `--config`.

`input.video_path` in `Config.toml` can now point to either a single video file or a folder of videos for batch processing. When you use a folder, the pipeline runs once per video and creates a separate output directory for each clip.

## Key Features

- 70 MHR70 keypoints including body, hands, and feet
- `sam3`, `vitdet`, `yolo11`, or `none` detector options
- MoGe2 FOV estimation for improved camera geometry
- Single-person selection and tracking enabled by default
- Manual support-surface ROI selection for treadmill or elevated-contact scenes
- Contact-aware ground alignment for jumps and landing-sensitive motion
- Hybrid vertical translation that blends support-plane cues and `pred_cam_t`
- OpenSim IK with centralized runtime foot markers including heel, big toe, and small toe
- Optional post-IK stance-phase foot snap to reduce visible landing hover
- Two-stage workflow for fast export iteration without re-running inference

## MHR70 Keypoint Index Map

The canonical 70-keypoint order is defined in [`src/keypoint_converter.py`](src/keypoint_converter.py).

These are three different layers in the current pipeline:

| Layer | Count | Meaning |
|------|------:|---------|
| Raw MHR70 keypoints | 70 | Direct SAM3D Body outputs before marker conversion |
| Exported TRC markers | 73 | 70 direct marker names plus 3 derived markers |
| OpenSim IK runtime markers | 28 | The subset actually used by the IK task set |

The tables below show the 70 raw MHR70 keypoints and the default direct marker names used during TRC export. These tables do not mean that OpenSim IK uses all 70 markers.

### Body and Feet

| Index | MHR70 Name | Direct Export Marker Name |
|------:|------------|----------------|
| 0 | `nose` | `Nose` |
| 1 | `left_eye` | `LEye` |
| 2 | `right_eye` | `REye` |
| 3 | `left_ear` | `LEar` |
| 4 | `right_ear` | `REar` |
| 5 | `left_shoulder` | `LShoulder` |
| 6 | `right_shoulder` | `RShoulder` |
| 7 | `left_elbow` | `LElbow` |
| 8 | `right_elbow` | `RElbow` |
| 9 | `left_hip` | `LHip` |
| 10 | `right_hip` | `RHip` |
| 11 | `left_knee` | `LKnee` |
| 12 | `right_knee` | `RKnee` |
| 13 | `left_ankle` | `LAnkle` |
| 14 | `right_ankle` | `RAnkle` |
| 15 | `left_big_toe` | `LBigToe` |
| 16 | `left_small_toe` | `LSmallToe` |
| 17 | `left_heel` | `LHeel` |
| 18 | `right_big_toe` | `RBigToe` |
| 19 | `right_small_toe` | `RSmallToe` |
| 20 | `right_heel` | `RHeel` |

### Right Hand

| Index | MHR70 Name | Direct Export Marker Name |
|------:|------------|----------------|
| 21 | `right_thumb_tip` | `RThumbTip` |
| 22 | `right_thumb_first_joint` | `RThumb1` |
| 23 | `right_thumb_second_joint` | `RThumb2` |
| 24 | `right_thumb_third_joint` | `RThumb3` |
| 25 | `right_index_tip` | `RIndexTip` |
| 26 | `right_index_first_joint` | `RIndex1` |
| 27 | `right_index_second_joint` | `RIndex2` |
| 28 | `right_index_third_joint` | `RIndex3` |
| 29 | `right_middle_tip` | `RMiddleTip` |
| 30 | `right_middle_first_joint` | `RMiddle1` |
| 31 | `right_middle_second_joint` | `RMiddle2` |
| 32 | `right_middle_third_joint` | `RMiddle3` |
| 33 | `right_ring_tip` | `RRingTip` |
| 34 | `right_ring_first_joint` | `RRing1` |
| 35 | `right_ring_second_joint` | `RRing2` |
| 36 | `right_ring_third_joint` | `RRing3` |
| 37 | `right_pinky_tip` | `RPinkyTip` |
| 38 | `right_pinky_first_joint` | `RPinky1` |
| 39 | `right_pinky_second_joint` | `RPinky2` |
| 40 | `right_pinky_third_joint` | `RPinky3` |
| 41 | `right_wrist` | `RWrist` |

### Left Hand

| Index | MHR70 Name | Direct Export Marker Name |
|------:|------------|----------------|
| 42 | `left_thumb_tip` | `LThumbTip` |
| 43 | `left_thumb_first_joint` | `LThumb1` |
| 44 | `left_thumb_second_joint` | `LThumb2` |
| 45 | `left_thumb_third_joint` | `LThumb3` |
| 46 | `left_index_tip` | `LIndexTip` |
| 47 | `left_index_first_joint` | `LIndex1` |
| 48 | `left_index_second_joint` | `LIndex2` |
| 49 | `left_index_third_joint` | `LIndex3` |
| 50 | `left_middle_tip` | `LMiddleTip` |
| 51 | `left_middle_first_joint` | `LMiddle1` |
| 52 | `left_middle_second_joint` | `LMiddle2` |
| 53 | `left_middle_third_joint` | `LMiddle3` |
| 54 | `left_ring_tip` | `LRingTip` |
| 55 | `left_ring_first_joint` | `LRing1` |
| 56 | `left_ring_second_joint` | `LRing2` |
| 57 | `left_ring_third_joint` | `LRing3` |
| 58 | `left_pinky_tip` | `LPinkyTip` |
| 59 | `left_pinky_first_joint` | `LPinky1` |
| 60 | `left_pinky_second_joint` | `LPinky2` |
| 61 | `left_pinky_third_joint` | `LPinky3` |
| 62 | `left_wrist` | `LWrist` |

### Extra Anatomical Landmarks

| Index | MHR70 Name | Direct Export Marker Name |
|------:|------------|----------------|
| 63 | `left_olecranon` | `LOlecranon` |
| 64 | `right_olecranon` | `ROlecranon` |
| 65 | `left_cubital_fossa` | `LCubitalFossa` |
| 66 | `right_cubital_fossa` | `RCubitalFossa` |
| 67 | `left_acromion` | `LAcromion` |
| 68 | `right_acromion` | `RAcromion` |
| 69 | `neck` | `Neck` |

### Derived TRC Markers

The canonical export path adds three derived markers on top of the 70 direct markers:

| Derived Marker | Meaning |
|----------------|---------|
| `PelvisCenter` | Midpoint derived from left and right hip landmarks |
| `Thorax` | Upper-torso proxy used for OpenSim export and debugging |
| `SpineMid` | Mid-spine proxy derived from torso landmarks |

This is why the default TRC export currently writes 73 markers.

### OpenSim IK Runtime Marker Subset

OpenSim IK does not use all 73 exported TRC markers. The runtime task set currently uses the following 28-marker subset:

| Group | Runtime IK Markers |
|-------|--------------------|
| Head | `Nose`, `LEye`, `REye`, `LEar`, `REar` |
| Torso | `Neck`, `LShoulder`, `RShoulder` |
| Arms and hands | `LElbow`, `RElbow`, `LWrist`, `RWrist`, `LIndex3`, `RIndex3`, `LMiddleTip`, `RMiddleTip` |
| Legs | `LHip`, `RHip`, `LKnee`, `RKnee`, `LAnkle`, `RAnkle` |
| Feet | `LBigToe`, `LSmallToe`, `LHeel`, `RBigToe`, `RSmallToe`, `RHeel` |

In short:

- `70` raw MHR70 keypoints come from SAM3D Body
- `73` markers are written to the TRC export path
- `28` markers are used by the current OpenSim IK runtime task set

The default source of truth for:

- direct marker names
- derived marker definitions
- runtime IK marker subset order
- runtime IK weight overrides

is `config/marker_mapping.yaml`.

## Recommended Workflow

### Full Pipeline

```bash
sam3d-opensim --config Config.toml
```

Equivalent legacy command:

```bash
python run_full_pipeline.py --config Config.toml
```

Batch example:

```toml
[input]
video_path = "D:/clips/session_01"

[output]
directory = "D:/clips/session_01_outputs"
```

### Full Pipeline With Manual Support Surface

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation \
    --support-surface-mode manual_roi \
    --vertical-translation-mode hybrid_support_plane \
    --post-ik-foot-snap stance_only
```

### Two-Stage Workflow

Stage 1, slow, run once:

```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

Stage 1 with mesh overlay video and Blender-importable mesh sequence:

```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2 \
    --save-mesh-video --save-mesh-sequence
```

Stage 2, fast, re-run as needed:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation \
    --ground-alignment-mode contact_aware \
    --vertical-translation-mode hybrid_support_plane \
    --save_graph
```

Stage 2 with Pose2Sim marker augmentation:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --ik-backend pose2sim_augmented --skip-fbx
```

### CPU Inference

`run_full_pipeline.py` does not expose a `--device` flag. For CPU inference, use the two-stage workflow:

```bash
python run_inference.py --input video.mp4 --device cpu
python run_export.py --input output_dir/video_outputs.json --height 1.69
```

## Common Flags

### Inference / Full Pipeline

| Flag | Meaning | Default |
|------|---------|---------|
| `--input, -i` | Input video path | Required |
| `--height` | Subject height in meters | `1.75` |
| `--mass` | Subject mass in kilograms | `70.0` |
| `--output, -o` | Output directory | auto-generated |
| `--fps` | Target processing FPS | `30.0` |
| `--detector` | `vitdet`, `yolo11`, `sam3`, `none` | `vitdet` |
| `--segmentor` | `sam2`, `none` | `none` |
| `--fov` | `moge2`, `none` | `moge2` |
| `--single_person` | Prompt once and track a single person | `true` |
| `--support-surface-mode` | `auto`, `manual_roi` | config default (`auto`) |
| `--use-mask` | Use segmentation masks when available | `false` |
| `--save-mesh-video` | Save Stage 1 mesh overlay video under `mesh_vis/overlay.mp4` | `false` |
| `--save-mesh-sequence` | Save per-frame mesh files under `mesh_export/` | `false` |
| `--mesh-sequence-format` | `ply`, `obj` for `--save-mesh-sequence` | `ply` |

### Export / Motion Reconstruction

| Flag | Meaning | Default |
|------|---------|---------|
| `--global-translation` | Apply `cam_t` translation | `false` |
| `--ik-backend` | `direct_opensim`, `pose2sim_augmented` | `direct_opensim` |
| `--smooth` | Butterworth cutoff frequency in Hz | `6.0` |
| `--ground-alignment-mode` | `auto`, `contact_aware`, `per_frame_snap` | `auto` |
| `--vertical-translation-mode` | `auto`, `legacy_xz_only`, `hybrid_support_plane` | `auto` |
| `--post-ik-foot-snap` | `off`, `auto`, `stance_only` | `off` |
| `--save_graph` | Save TRC coordinate and MOT angle plots under `graphs/` | `false` |
| `--skip-ik` | Skip OpenSim IK | `false` |
| `--skip-fbx` | Skip Blender FBX export | `false` |

## Output Files

Typical output directory:

```text
output_YYYYMMDD_HHMMSS_<video>/
├── frames/                        # Extracted video frames
├── video_outputs.json             # Canonical SAM3D per-frame outputs
├── inference_meta.json            # FPS, video info, selection, scene-ground metadata
├── mesh_vis/overlay.mp4           # Stage 1 mesh overlay video when --save-mesh-video is enabled
├── mesh_export/frame_*.ply        # Stage 1 per-frame mesh files when --save-mesh-sequence is enabled
├── post_ik_contact_meta.json      # Stance/flight metadata used by post-IK correction
├── markers_*.trc                  # OpenSim marker trajectories
├── *_ik.mot                       # OpenSim IK result
├── *_ik_raw.mot                   # Raw IK result when post-IK foot snap is enabled
├── *_foot_snap_report.json        # Post-IK correction report when enabled
├── *_model.osim                   # OpenSim model with runtime markers
├── graphs/coords/*.png            # TRC marker coordinate plots when --save_graph is enabled
├── graphs/angles/*.png            # MOT angle plots when --save_graph is enabled and IK runs
└── *.fbx                          # Animated skeleton export
```

Notes:

- `run_inference.py` writes `video_outputs.json` and `inference_meta.json`
- `run_inference.py` can additionally write `mesh_vis/overlay.mp4` and `mesh_export/` as Stage 1 sidecar outputs
- mesh sidecars are generated from a visualization-quality full-refresh pass so they can stay closer to the official SAM3D Body demo semantics
- `run_export.py` consumes those files and writes the downstream artifacts
- `run_pipeline.py` may additionally write `keypoints_raw.json` and `processing_report.json`
- `run_full_pipeline.py` stays a thin subprocess wrapper and does not add extra in-process reporting artifacts
- with `--save_graph`, coordinate plots are written after TRC export; angle plots are written only when IK produces a MOT file
- with `--ik-backend pose2sim_augmented`, Stage 2 also creates `pose2sim_trial/pose-3d` and `pose2sim_trial/kinematics` and runs Pose2Sim marker augmentation before LSTM kinematics

## Practical Guidance

- Use `--detector sam3 --fov moge2` for the best overall Stage 1 quality.
- Use `--save-mesh-video` when you want a quick sanity-check video of raw SAM3D mesh quality before any OpenSim export.
- Use `--save-mesh-sequence --mesh-sequence-format ply` when you want per-frame meshes that Blender can import directly.
- Expect `--save-mesh-video` to be slower than normal inference because it bypasses the tracked-bbox fast path and focal-length cache for the mesh sidecar pass.
- Use `--global-translation` only when subject motion through space matters.
- Use `--support-surface-mode manual_roi` when the contact plane is not the room floor, for example treadmill, box, or platform scenes.
- Use `--ground-alignment-mode contact_aware` for jumps or flight phases.
- Use `--vertical-translation-mode hybrid_support_plane` when support-surface grounding is available.
- Use `--post-ik-foot-snap stance_only` if the IK result still shows visible stance-phase hover.
- Use `--ik-backend pose2sim_augmented` only when you explicitly want the Pose2Sim marker augmentation + LSTM IK path. The default `direct_opensim` path keeps the repo's current runtime marker/task flow.

## Environments

| Environment | Purpose |
|-------------|---------|
| `sam_3d_body` | SAM3D Body inference, SAM3, and MoGe2 |
| `Pose2Sim` | OpenSim inverse kinematics and post-IK MOT correction |

## External Dependencies

- SAM3D Body
- SAM3
- MoGe2
- OpenSim via Pose2Sim
- Blender 4.5+ or any auto-detected Blender executable under `C:\Program Files\Blender Foundation\Blender *\blender.exe`

## Verification

After CLI or orchestration changes, run:

```bash
python run_inference.py --help
python run_export.py --help
python run_full_pipeline.py --help
```

For export-stage changes, the fastest smoke test is:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 --skip-ik --skip-fbx
```

For the Pose2Sim augmentation backend, the fastest smoke test is:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --ik-backend pose2sim_augmented --skip-fbx
```

`run_pipeline.py --visualize` is currently a deprecated compatibility flag and does not generate extra outputs.

Use `--save-mesh-video` or `--save-mesh-sequence` instead when you want real Stage 1 mesh artifacts.

## Documentation

- [Full Documentation](docs/FULL_DOCUMENTATION.md)

## License

This project is provided for research and educational purposes.

## Acknowledgments

- [SAM3D Body](https://github.com/facebookresearch/sam-3d-body)
- [SAM3](https://github.com/facebookresearch/sam3)
- [Pose2Sim](https://github.com/perfanalytics/pose2sim)
- [OpenSim](https://opensim.stanford.edu/)
