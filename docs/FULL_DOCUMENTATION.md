# SAM3D Body to OpenSim Pipeline - Complete Documentation

**Version:** 2.0.0
**Created:** 2026-02-06
**Last Updated:** 2026-02-06
**Purpose:** Convert video to OpenSim motion data and FBX animation using SAM3D Body 3D pose estimation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Usage](#5-usage)
6. [Two-Stage Workflow](#6-two-stage-workflow)
7. [Global Translation Tracking](#7-global-translation-tracking)
8. [SAM3D Components](#8-sam3d-components)
9. [File Reference](#9-file-reference)
10. [MHR70 Marker Mapping](#10-mhr70-marker-mapping)
11. [OpenSim Model & IK](#11-opensim-model--ik)
12. [FBX Export (Blender)](#12-fbx-export-blender)
13. [Coordinate Systems](#13-coordinate-systems)
14. [Troubleshooting](#14-troubleshooting)
15. [API Reference](#15-api-reference)
16. [Test Results](#16-test-results)
17. [Changelog](#17-changelog)

---

## 1. Overview

### Purpose

This pipeline replicates the MotionBERT-OpenSim approach but uses **SAM3D Body** (from Meta/Facebook Research) for 3D pose estimation. SAM3D Body provides:

- Direct 3D pose estimation (no 2D→3D lifting)
- 70 keypoints (MHR70 format) including hands and feet
- Mesh vertices (6890 points)
- Shape and pose parameters
- Camera translation (cam_t) for global positioning

### Key Features (v2.0)

- **SAM3 Detector**: Use Segment Anything 3 for person detection (aligns with SAM3D online playground)
- **MoGe2 FOV Estimation**: Accurate focal length estimation for better 3D reconstruction
- **Global Translation Tracking**: Track walking/running movement using cam_t
- **Two-Stage Workflow**: Separate inference (slow) from export (fast) for rapid iteration
- **Multiple Output Formats**: TRC, MOT, FBX, JSON

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SAM3DBodyToOpenSim v2.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │  Video  │───▶│ SAM3 Detector│───▶│ SAM3D Body  │───▶│ MHR70 3D       │  │
│  │  Input  │    │ (person bbox)│    │ (3D pose)   │    │ Keypoints+cam_t│  │
│  └─────────┘    └──────────────┘    └─────────────┘    └───────┬────────┘  │
│                        │                                        │           │
│                        ▼                                        │           │
│                 ┌──────────────┐                               │           │
│                 │ MoGe2 FOV    │ (focal length on 1st frame)   │           │
│                 └──────────────┘                               │           │
│                                                                 │           │
│                     ┌───────────────────────────────────────────┘           │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    video_outputs.json (SAM3D format)                  │  │
│  │  - Per-frame: bbox, focal_length, pred_keypoints_3d, pred_cam_t      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Post-Processing Pipeline                         │  │
│  │  1. Coordinate transformation (Camera → OpenSim)                     │  │
│  │  2. Height-based scaling                                              │  │
│  │  3. Global translation from cam_t (optional)                         │  │
│  │  4. Ground alignment                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                     │                                                       │
│         ┌───────────┼───────────┬───────────────┐                          │
│         ▼           ▼           ▼               ▼                          │
│  ┌────────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐                    │
│  │ TRC Export │ │OpenSim │ │ Blender  │ │ Processing │                    │
│  │ (18 marks) │ │ IK     │ │ FBX      │ │ Report     │                    │
│  └─────┬──────┘ └───┬────┘ └────┬─────┘ └────────────┘                    │
│        │            │           │                                          │
│        ▼            ▼           ▼                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                   │
│  │ .trc     │ │ .mot     │ │ .fbx     │                                   │
│  │ markers  │ │ 40 DOF   │ │ skeleton │                                   │
│  └──────────┘ └──────────┘ └──────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Output Files Summary

| Stage | Output | Format | Description |
|-------|--------|--------|-------------|
| Inference | `video_outputs.json` | JSON | SAM3D format with keypoints, cam_t, focal_length |
| Inference | `inference_meta.json` | JSON | Video info, FPS, frame count |
| Post-process | `markers_*.trc` | TRC | 18 OpenSim-compatible markers |
| OpenSim IK | `*_ik.mot` | MOT | 40 joint angles in degrees |
| OpenSim | `*_model.osim` | OSIM | Scaled model with markers |
| Blender | `*.fbx` | FBX | Animated skeleton for 3D software |

### Comparison with MotionBERT-OpenSim

| Aspect | MotionBERT-OpenSim | SAM3D Body (This Pipeline) |
|--------|-------------------|----------------------------|
| Input | Video → 2D COCO17 → 3D H36M | Video → Direct 3D (MHR70) |
| Skeleton | 17 joints (H36M) | 70 keypoints (MHR70) |
| Hands/Feet | No | Yes (detailed) |
| Mesh | No | Yes (6890 vertices) |
| Global Position | No | Yes (cam_t tracking) |
| FOV Estimation | No | Yes (MoGe2) |

---

## 2. Architecture

### Project Structure

```
C:\Sam3DBodyToOpenSim\
├── config/
│   ├── config.yaml              # Main pipeline configuration
│   └── marker_mapping.yaml      # MHR70 → OpenSim marker mapping
├── models/
│   ├── Model_Simple_MHR70.osim  # Simple kinematic OpenSim model
│   └── Markers_MHR70.xml        # Marker definitions for Pose2Sim
├── src/
│   ├── __init__.py              # Package initialization
│   ├── sam3d_inference.py       # SAM3D Body model wrapper
│   ├── keypoint_converter.py    # MHR70 → OpenSim conversion
│   ├── coordinate_transform.py  # Camera → OpenSim coords + global translation
│   ├── post_processing.py       # Bone normalization, smoothing
│   ├── trc_exporter.py          # TRC file generation
│   └── opensim_ik.py            # Inverse kinematics solver
├── utils/
│   ├── __init__.py              # Utilities package
│   ├── video_utils.py           # Frame extraction
│   └── io_utils.py              # Config/JSON I/O
├── docs/
│   └── FULL_DOCUMENTATION.md    # This file
├── scripts/
│   └── export_fbx_blender.py    # Blender FBX export script
├── videos/                      # Input videos
├── run_inference.py             # Stage 1: SAM3D inference → JSON (slow)
├── run_export.py                # Stage 2: JSON → TRC/MOT/FBX (fast)
├── run_pipeline.py              # Combined TRC pipeline
├── run_full_pipeline.py         # Full pipeline (TRC + IK + FBX)
├── test_imports.py              # Test SAM3/MoGe2 imports
├── test_pipeline.py             # Unit tests
├── requirements.txt             # Python dependencies
└── README.md                    # Quick start guide
```

### Conda Environments

| Environment | Python | Purpose | Key Packages |
|-------------|--------|---------|--------------|
| `sam_3d_body` | 3.11 | SAM3D Body inference | PyTorch 2.9.1+CUDA, sam-3d-body, sam3, moge |
| `Pose2Sim` | 3.11 | OpenSim IK | OpenSim 4.5.2, Pose2Sim |

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `sam3d_inference.py` | Load SAM3D Body, SAM3 detector, MoGe2 FOV; process frames |
| `keypoint_converter.py` | Map MHR70 indices to OpenSim marker names |
| `coordinate_transform.py` | Transform coords, apply global translation from cam_t |
| `post_processing.py` | Normalize bone lengths, optional smoothing |
| `trc_exporter.py` | Generate TRC files for OpenSim |
| `opensim_ik.py` | Run inverse kinematics via Pose2Sim/OpenSim |

---

## 3. Installation

### Prerequisites

- Windows 10/11
- Python 3.11
- CUDA-capable GPU (8GB+ VRAM recommended)
- Anaconda or Miniconda
- SAM3D Body installed at `C:\Sam3dBody\sam-3d-body`
- SAM3 installed at `C:\Sam3dBody\sam3`
- Blender 5.0+ (for FBX export)

### Step 1: SAM3D Body Environment

```bash
# Activate SAM3D Body environment
conda activate sam_3d_body

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Step 2: Install SAM3 (Segment Anything 3)

SAM3 is required for the `--detector sam3` option.

```bash
conda activate sam_3d_body

# Clone SAM3 repository (if not already done)
cd C:\Sam3dBody
git clone https://github.com/facebookresearch/sam3.git

# Install SAM3
cd sam3
pip install -e .

# Install SAM3 dependencies
pip install decord psutil ftfy regex triton-windows
```

**SAM3 Path:** `C:\Sam3dBody\sam3\sam3` (nested structure)

### Step 3: Install MoGe2 (FOV Estimation)

MoGe2 should already be installed with SAM3D Body. Verify:

```bash
python -c "from moge.model.v2 import MoGeModel; print('MoGe2 OK')"
```

### Step 4: Verify All Imports

```bash
cd C:\Sam3DBodyToOpenSim
python test_imports.py
```

Expected output:
```
1. Testing SAM3 import...
   [OK] SAM3 imports successful
2. Testing MoGe2 import...
   [OK] MoGe2 imports successful
3. Testing FOVEstimator from SAM3D Body...
   [OK] FOVEstimator import successful
4. Testing HumanDetector from SAM3D Body...
   [OK] HumanDetector import successful
```

### Step 5: Pose2Sim Environment (for OpenSim IK)

```bash
conda activate Pose2Sim
python -c "import opensim; print(opensim.GetVersion())"  # Should print 4.5.x
```

### Step 6: Verify Blender

```bash
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --version
```

---

## 4. Configuration

### Main Configuration (config/config.yaml)

```yaml
input:
  video_path: null  # Set via CLI
  fps: 30           # Target FPS for extraction

subject:
  height: 1.75      # Subject height in meters
  mass: 70          # Subject mass in kg

sam3d:
  sam3d_root: "C:/Sam3dBody/sam-3d-body"
  checkpoint: "C:/Sam3dBody/checkpoints/sam-3d-body-dinov3/model.ckpt"
  mhr_path: "C:/Sam3dBody/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

  # Component options (can be overridden via CLI)
  detector_name: "vitdet"   # Options: vitdet, yolo11, sam3, or null
  segmentor_name: null      # Options: sam2 or null
  fov_name: "moge2"         # Options: moge2 or null (RECOMMENDED: moge2)

  device: "cuda"
  bbox_threshold: 0.8
  use_mask: false
  inference_type: "full"    # "full", "body", or "hand"

processing:
  smooth_filter: true       # Butterworth low-pass filter
  filter_cutoff: 6          # Hz - cutoff frequency
  normalize_bones: true
  correct_forward_lean: true

opensim:
  model: "models/Model_Simple_MHR70.osim"
  markers_xml: "models/Markers_MHR70.xml"
  ik_accuracy: 1.0e-5

output:
  save_trc: true
  save_mot: true
  save_json: true
```

---

## 5. Usage

### Quick Start (Recommended)

```bash
conda activate sam_3d_body
cd C:\Sam3DBodyToOpenSim

# Full pipeline with SAM3 detector, MoGe2 FOV, and global translation
python run_full_pipeline.py --input videos/walking.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input video file | Required |
| `--height` | Subject height (meters) | 1.75 |
| `--mass` | Subject mass (kg) | 70.0 |
| `--output, -o` | Output directory | Auto |
| `--fps` | Target FPS | 30 |
| `--detector` | Human detector: vitdet, yolo11, sam3, none | vitdet |
| `--segmentor` | Segmentor: sam2, none | none |
| `--fov` | FOV estimator: moge2, none | moge2 |
| `--use-mask` | Use segmentation mask | false |
| `--global-translation` | Track global movement from cam_t | false |
| `--smooth` | Smoothing cutoff frequency in Hz (0 to disable) | 6.0 |
| `--skip-inference` | Skip SAM3D (use existing data) | false |
| `--skip-ik` | Skip OpenSim IK | false |
| `--skip-fbx` | Skip FBX export | false |

### Best Quality Configuration

For maximum quality results:

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 \
    --fov moge2 \
    --global-translation
```

This uses:
- **SAM3**: Best person detection (text prompt "person")
- **MoGe2**: Accurate focal length estimation (critical for 3D accuracy)
- **Global translation**: Tracks walking/running movement using cam_t

---

## 6. Two-Stage Workflow

For faster iteration, the pipeline can be split into two stages:

### Stage 1: Inference (Slow, run once)

```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

**Output:** `output_dir/video_outputs.json` (SAM3D format)

**Time:** ~1.3 sec/frame (depends on GPU)

### Stage 2: Export (Fast, iterate on settings)

```bash
python run_export.py --input output_dir/video_outputs.json \
    --height 1.69 --global-translation
```

**Output:** TRC, MOT, FBX files

**Time:** ~10 seconds for 1000+ frames

### Benefits

| Stage | Time (1136 frames) | Purpose |
|-------|-------------------|---------|
| Inference | ~25 min | SAM3D Body 3D pose estimation (run once) |
| Export | ~10 sec | Coordinate transform, TRC, IK, FBX (iterate) |

### Using Existing SAM3D Outputs

You can also use outputs from the SAM3D Body demo directly:

```bash
python run_export.py --input C:\Sam3dBody\sam-3d-body\outputs\my_video\video_outputs.json \
    --height 1.75 --global-translation
```

---

## 7. Global Translation Tracking

### Overview

Global translation tracking uses `cam_t` (camera translation) from SAM3D Body to track the subject's movement through 3D space. This is essential for:

- Walking/running gait analysis
- Any video where the subject moves significantly
- Matching real-world movement distances

### How It Works

1. **MoGe2 FOV Estimation**: Estimates camera focal length from the first frame
2. **cam_t Extraction**: SAM3D Body outputs `pred_cam_t = [x, y, z]` per frame
3. **Coordinate Transform**: cam_t is transformed from camera space to OpenSim space
4. **Smoothing**: Light smoothing applied to reduce frame-to-frame jitter
5. **Relative Translation**: Movement is relative to first frame (starts at origin)

### cam_t Coordinate System

**Camera Space (SAM3D Body):**
- X: Right
- Y: Down
- Z: Forward (depth)

**OpenSim Space (after transform):**
- X: Forward (anterior)
- Y: Up (superior)
- Z: Right (lateral)

### Enabling Global Translation

```bash
python run_full_pipeline.py --input walking.mp4 --height 1.69 \
    --fov moge2 --global-translation
```

**Important:** MoGe2 FOV estimation (`--fov moge2`) is highly recommended when using global translation, as accurate focal length is critical for correct cam_t values.

### Implementation Details

The global translation is applied in `src/coordinate_transform.py`:

```python
def _apply_global_translation(self, keypoints, camera_translation, height_scale):
    # Transform cam_t from camera space to OpenSim space
    cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T

    # Scale by height normalization factor
    cam_t_opensim = cam_t_opensim * height_scale

    # Apply smoothing to reduce jitter
    cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)

    # Compute relative translation (from first frame)
    first_frame_t = cam_t_smoothed[0].copy()

    # Apply X (forward) and Z (lateral) translation
    for i in range(num_frames):
        delta_t = cam_t_smoothed[i] - first_frame_t
        keypoints[i, :, 0] += delta_t[0]  # X (forward)
        keypoints[i, :, 2] += delta_t[2]  # Z (lateral)
```

---

## 8. SAM3D Components

### Available Components

| Component | Options | Description |
|-----------|---------|-------------|
| **Detector** | vitdet, yolo11, sam3, none | Person bounding box detection |
| **Segmentor** | sam2, none | Person segmentation mask |
| **FOV Estimator** | moge2, none | Camera focal length estimation |

### SAM3 Detector

SAM3 (Segment Anything 3) uses a text prompt ("person") to detect humans. This aligns with the SAM3D online playground.

**Advantages:**
- State-of-the-art detection
- Handles challenging poses
- Aligns with official SAM3D demo

**Requirements:**
- SAM3 installed at `C:\Sam3dBody\sam3`
- Dependencies: decord, psutil, ftfy, regex, triton-windows

**Usage:**
```bash
python run_full_pipeline.py --input video.mp4 --detector sam3
```

### MoGe2 FOV Estimator

MoGe2 estimates camera intrinsics (focal length) from a single image. This is critical for:

- Accurate 3D reconstruction
- Correct cam_t values for global translation
- Proper depth scaling

**How it works:**
1. Runs on the **first frame only** (cached for efficiency)
2. Estimates vertical focal length
3. Applied to all subsequent frames

**Usage:**
```bash
python run_full_pipeline.py --input video.mp4 --fov moge2
```

### Component Combinations

| Configuration | Quality | Speed | Use Case |
|---------------|---------|-------|----------|
| sam3 + moge2 | ⭐⭐⭐⭐⭐ | Slowest | Best quality, global translation |
| vitdet + moge2 | ⭐⭐⭐⭐ | Fast | Good quality, global translation |
| vitdet only | ⭐⭐⭐ | Fastest | Quick preview, no global translation |

---

## 9. File Reference

### Input Files

| File | Description |
|------|-------------|
| `video.mp4` | Input video (any format supported by OpenCV) |

### Output Files

| File | Description |
|------|-------------|
| `frames/` | Extracted video frames (PNG) |
| `video_outputs.json` | SAM3D format: per-frame keypoints, cam_t, focal_length |
| `inference_meta.json` | Video metadata: FPS, frame count, dimensions |
| `keypoints_raw.json` | Legacy format: raw MHR70 keypoints |
| `markers_*.trc` | OpenSim marker trajectories (18 markers) |
| `markers_*_ik.mot` | Joint angles (40 DOF) |
| `markers_*_model.osim` | Scaled OpenSim model with markers |
| `markers_*.fbx` | Animated skeleton for Blender/Unity |
| `processing_report.json` | Timing and processing statistics |

### video_outputs.json Format

```json
[
  {
    "frame": "frame_000001.png",
    "outputs": [
      {
        "bbox": [x1, y1, x2, y2],
        "focal_length": 1234.56,
        "pred_keypoints_3d": [[x, y, z], ...],  // 70 keypoints
        "pred_cam_t": [tx, ty, tz]
      }
    ]
  },
  ...
]
```

---

## 10. MHR70 Marker Mapping

### Body Markers (17)

| Index | MHR70 Name | OpenSim Marker |
|-------|------------|----------------|
| 0 | nose | Nose |
| 1 | left_eye | LEye |
| 2 | right_eye | REye |
| 3 | left_ear | LEar |
| 4 | right_ear | REar |
| 5 | left_shoulder | LShoulder |
| 6 | right_shoulder | RShoulder |
| 7 | left_elbow | LElbow |
| 8 | right_elbow | RElbow |
| 9 | left_hip | LHip |
| 10 | right_hip | RHip |
| 11 | left_knee | LKnee |
| 12 | right_knee | RKnee |
| 13 | left_ankle | LAnkle |
| 14 | right_ankle | RAnkle |
| 41 | right_wrist | RWrist |
| 62 | left_wrist | LWrist |

### Feet Markers (6)

| Index | MHR70 Name | OpenSim Marker |
|-------|------------|----------------|
| 15 | left_big_toe | LBigToe |
| 16 | left_small_toe | LSmallToe |
| 17 | left_heel | LHeel |
| 18 | right_big_toe | RBigToe |
| 19 | right_small_toe | RSmallToe |
| 20 | right_heel | RHeel |

### Markers Used for IK (22)

The OpenSim IK uses these 22 markers with weights:

| Marker | Weight | Body Segment |
|--------|--------|--------------|
| Nose | 0.8 | head |
| LEye | 0.4 | head |
| REye | 0.4 | head |
| LEar | 0.6 | head |
| REar | 0.6 | head |
| Neck | 1.0 | torso |
| LShoulder | 1.0 | torso |
| RShoulder | 1.0 | torso |
| LElbow | 1.0 | humerus_l |
| RElbow | 1.0 | humerus_r |
| LWrist | 1.0 | radius_l |
| RWrist | 1.0 | radius_r |
| LIndex3 | 0.6 | radius_l |
| RIndex3 | 0.6 | radius_r |
| LMiddleTip | 0.5 | radius_l |
| RMiddleTip | 0.5 | radius_r |
| LHip | 1.0 | pelvis |
| RHip | 1.0 | pelvis |
| LKnee | 1.0 | femur_l |
| RKnee | 1.0 | femur_r |
| LAnkle | 1.0 | tibia_l |
| RAnkle | 1.0 | tibia_r |

### Hand Markers for Arm Rotation

Four hand markers are used to improve forearm pronation/supination tracking:

- **LIndex3/RIndex3**: Index knuckle (weight 0.6)
- **LMiddleTip/RMiddleTip**: Middle fingertip (weight 0.5)

These are attached to the radius (forearm) segment in OpenSim, helping constrain the internal/external rotation of the arms.

---

## 11. OpenSim Model & IK

### Model: Pose2Sim Simple Model

- **DOF:** 40 degrees of freedom
- **Bodies:** 16 segments
- **Source:** `Pose2Sim/OpenSim_Setup/Model_Pose2Sim_simple.osim`

### IK Configuration

- **Accuracy:** 1e-5
- **Constraint Weight:** 20
- **Markers:** 18 (COCO17 + Neck)

### IK Output (.mot file)

Joint angles in degrees for 40 DOF including:
- Pelvis (6 DOF): tx, ty, tz, rx, ry, rz
- Spine/Torso (3 DOF)
- Neck/Head (3 DOF)
- Shoulders (6 DOF)
- Elbows (2 DOF)
- Hips (6 DOF)
- Knees (2 DOF)
- Ankles (4 DOF)

---

## 12. FBX Export (Blender)

### Requirements

- Blender 5.0+ installed at `C:\Program Files\Blender Foundation\Blender 5.0\`

### Skeleton Structure

18 bones matching the TRC markers:
- Root → Pelvis → Spine → Neck → Head
- Shoulders → Elbows → Wrists
- Hips → Knees → Ankles

### Usage in 3D Software

The FBX file can be imported into:
- Blender
- Maya
- Unity
- Unreal Engine

---

## 13. Coordinate Systems

### SAM3D Body (Camera Space)

```
        Y (down)
        |
        |
        +-----> X (right)
       /
      /
     Z (forward/depth)
```

### OpenSim (World Space)

```
        Y (up)
        |
        |
        +-----> Z (right)
       /
      /
     X (forward)
```

### Transformation Matrix

```python
CAMERA_TO_OPENSIM = np.array([
    [0,  0,  1],  # X_opensim = Z_camera (forward)
    [0, -1,  0],  # Y_opensim = -Y_camera (up)
    [1,  0,  0],  # Z_opensim = X_camera (right)
])
```

---

## 14. Troubleshooting

### SAM3 Import Error: "No module named 'sam3'"

**Solution:** Ensure SAM3 path is correct in `build_detector.py`:
```python
sam3_path = "C:/Sam3dBody/sam3/sam3"  # Note: nested structure
```

### SAM3 Import Error: "No module named 'triton'"

**Solution:** Install triton for Windows:
```bash
pip install triton-windows
```

### SAM3 Import Error: "No module named 'decord'" or 'psutil' or 'ftfy'

**Solution:** Install missing dependencies:
```bash
pip install decord psutil ftfy regex
```

### "No FOV estimator" message

**Cause:** MoGe2 failed to load or `--fov` argument not passed.

**Solution:**
1. Verify moge is installed: `python -c "from moge.model.v2 import MoGeModel"`
2. Explicitly pass `--fov moge2` argument

### Global translation jittery

**Cause:** cam_t values are noisy without proper FOV estimation.

**Solution:** Always use `--fov moge2` with `--global-translation`

### CUDA out of memory

**Solutions:**
1. Lower FPS: `--fps 15`
2. Use CPU (slower): Edit config.yaml `device: "cpu"`

### OpenSim IK fails

**Solutions:**
1. Check TRC file loads in OpenSim GUI
2. Verify marker names match expected format
3. Try `--skip-ik` to verify TRC generation

---

## 15. API Reference

### SAM3DInference

```python
from src.sam3d_inference import SAM3DInference

sam3d = SAM3DInference(
    sam3d_root="C:/Sam3dBody/sam-3d-body",
    checkpoint_path="...",
    mhr_path="...",
    device="cuda",
    detector_name="sam3",      # vitdet, yolo11, sam3, or None
    segmentor_name=None,       # sam2 or None
    fov_name="moge2",          # moge2 or None
    bbox_threshold=0.8,
    use_mask=False,
    inference_type="full",
)

# Process video
results = sam3d.process_video(frame_paths, progress=True)
keypoints_3d, valid_frames = sam3d.extract_keypoints_3d(results["frames"])
camera_params = sam3d.extract_camera_params(results["frames"])
```

### CoordinateTransformer

```python
from src.coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer(
    subject_height=1.69,
    units="mm",
)

keypoints_opensim = transformer.transform(
    keypoints_3d,
    camera_translation=cam_translations,  # From SAM3D Body
    center_pelvis=False,                   # Don't center if using global translation
    align_to_ground=True,
    apply_global_translation=True,         # Enable cam_t tracking
)
```

---

## 16. Test Results

### Pipeline Test (2026-02-06)

**Video:** aitor_garden_walk.mp4 (1920x1080, 37.8 sec)
**Subject Height:** 1.69 m
**Configuration:** SAM3 detector + MoGe2 FOV + Global translation

| Metric | Value |
|--------|-------|
| Total Frames | 1136 |
| Processing Time | ~25 min (inference) + 10 sec (export) |
| Inference Speed | ~1.3 sec/frame |
| IK Time | 4 seconds |
| Marker RMS Error | 0.04-0.07 m |
| Output Files | TRC ✅, MOT ✅, FBX ✅ |

---

## 17. Changelog

### v2.1.0 (2026-02-06)

**New Features:**
- **Butterworth Smoothing**: Configurable low-pass filter (`--smooth` flag, default 6 Hz)
- **Per-Frame Ground Alignment**: Feet always touch the floor (lowest foot point at Y=0)
- **Hand Markers for Arm Rotation**: LIndex3, RIndex3, LMiddleTip, RMiddleTip for better forearm pronation/supination

**Modified Files:**
- `run_export.py`: Added `--smooth` argument
- `run_pipeline.py`: Added `--smooth` argument
- `run_full_pipeline.py`: Added `--smooth` argument
- `src/coordinate_transform.py`: Per-frame ground alignment
- `src/post_processing.py`: Butterworth filter with configurable cutoff
- `config/config.yaml`: Enabled smoothing by default (6 Hz)

### v2.0.0 (2026-02-06)

**New Features:**
- **SAM3 Detector Integration**: Use Segment Anything 3 for person detection
- **MoGe2 FOV Estimation**: Accurate focal length for better 3D reconstruction
- **Global Translation Tracking**: Track movement using cam_t from SAM3D Body
- **Two-Stage Workflow**: Separate inference (slow) from export (fast)
- **video_outputs.json Export**: SAM3D-compatible format for interoperability

**New Files:**
- `run_inference.py`: Stage 1 - SAM3D inference only
- `run_export.py`: Stage 2 - Export to TRC/MOT/FBX
- `test_imports.py`: Verify SAM3/MoGe2 installation

**Modified Files:**
- `run_pipeline.py`: Added component options, JSON export
- `run_full_pipeline.py`: Added detector/fov/segmentor arguments
- `src/coordinate_transform.py`: Added cam_t global translation
- `src/sam3d_inference.py`: Added SAM3/MoGe2 component loading
- `C:\Sam3dBody\sam-3d-body\tools\build_detector.py`: Added SAM3 detector support
- `config/config.yaml`: Added fov_name default to moge2

**Dependencies Added:**
- triton-windows (for SAM3)
- decord, psutil, ftfy, regex (for SAM3)

### v1.0.0 (2026-02-06)

- Initial release
- Basic pipeline: Video → SAM3D Body → TRC → OpenSim IK → FBX
- VitDet detector support
- Pose2Sim model integration
